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
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = torch.cat((x, cat_tensor), dim=-1)
        x = self.fc2(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.out_size = num_inputs
        self.shared = SharedNetwork(hidden_dim)

        # Q1 architecture
        self.linear1 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        state = self.shared(state)
        xu = torch.cat([state, action], -1)

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

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        state = self.shared(state)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    # def sample(self, state):
    #     mean, log_std = self.forward(state)
    #     std = log_std.exp()
    #     normal = Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
    #     log_prob = log_prob.sum(-1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return action, log_prob, mean

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        pi_distribution = Normal(mean, std)
        pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
