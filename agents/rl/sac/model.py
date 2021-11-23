import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class SharedNetwork(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SharedNetwork, self).__init__()

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
        self.lstm = nn.LSTMCell(hidden_dim + 4, hidden_dim)

    def forward(self, obs, cat_tensor):
        obs, (hx, cx) = obs
        # print(obs.size())
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        # print(x.size())
        x = self.relu(self.conv6(x))
        # x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = torch.cat((x, cat_tensor), dim=1)
        hx, cx = self.lstm(x, (hx, cx))
        return hx, cx


class ValueNetwork(nn.Module):
    def __init__(self, input_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class QNetwork(nn.Module):
    def __init__(self, num_actions, input_dim=256):
        super(QNetwork, self).__init__()

        # Q1
        self.fc11 = nn.Linear(input_dim + num_actions, 64)
        self.fc12 = nn.Linear(64, num_actions)

        # Q2
        self.fc21 = nn.Linear(input_dim + num_actions, 64)
        self.fc22 = nn.Linear(64, num_actions)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x1 = self.relu(self.fc11(x))
        x1 = self.fc12(x1)

        x2 = self.relu(self.fc21(x))
        x2 = self.fc22(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class SAC(nn.Module):
    def __init__(self, num_actions, hidden_dim=256):
        super(SAC, self).__init__()
        self.shared_network = SharedNetwork(hidden_dim)
        self.value_network = ValueNetwork(hidden_dim)
        self.q_network = QNetwork(num_actions, hidden_dim)
        self.action_policy = GaussianPolicy(hidden_dim, num_actions, hidden_dim)

    def forward(self, x, lstm_state, cat_tensor):
        x = torch.reshape(x, (-1, 3, 400, 400))
        cat_tensor = torch.reshape(cat_tensor, (-1, 4))
        obs = (x, lstm_state)
        features, _ = self.shared_network(obs, cat_tensor)
        value = self.value_network(features)
        action, log_prob, mean = self.action_policy.sample(features)
        q_value1, q_value2 = self.q_network(features, action)

        return value, q_value1, q_value2, action, log_prob
