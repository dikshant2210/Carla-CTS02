import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SharedNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim=256):
        super(SharedNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        # input_shape = [None, 400, 400, 3]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(9, 9), stride=(3, 3))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(9, 9), stride=(1, 1))
        self.conv6 = nn.Conv2d(128, hidden_dim, kernel_size=(5, 5), stride=(1, 1))
        self.pool = nn.AdaptiveMaxPool2d((1, 1, hidden_dim))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(hidden_dim + 4, hidden_dim)

    def forward(self, obs):
        obs, cat_tensor = obs
        obs = obs.permute(0, 3, 1, 2)
        obs = F.normalize(obs)
        x1 = self.relu(self.conv1(obs))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x7 = self.relu(self.fc1(self.flatten(x6)))
        x7 = torch.cat((x7, cat_tensor), dim=-1)
        x8 = self.fc2(x7)
        if torch.isnan(x8).sum().item() > 0:
            print(torch.isnan(x1).sum().item(), torch.isnan(x2).sum().item(), torch.isnan(x3).sum().item(),
                  torch.isnan(x4).sum().item(), torch.isnan(x5).sum().item(), torch.isnan(x6).sum().item(),
                  torch.isnan(x7).sum().item(), torch.isnan(x8).sum().item())
        return x8


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.out_size = num_inputs
        self.shared = SharedNetwork(hidden_dim)

        # Q1 architecture
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        xu = self.shared(state)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.shared = SharedNetwork(hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        state = self.shared(state)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        out = self.out(x)
        # if torch.isnan(out).sum().item() > 0:
        #     print(torch.isnan(out).sum().item(), torch.isnan(x).sum().item(), torch.isnan(state).sum().item())
        return out

    def sample(self, state):
        out = self.forward(state)
        action_probs = F.softmax(out, dim=1)
        action_dist = torch.distributions.Categorical(probs=action_probs)
        actions = action_dist.sample().view(-1, 1)

        z = (action_probs == 0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, log_action_probs, action_probs

    def to(self, device):
        return super(GaussianPolicy, self).to(device)
