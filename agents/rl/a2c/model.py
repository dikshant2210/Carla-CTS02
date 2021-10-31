import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedNetwork(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SharedNetwork, self).__init__()

        # input_shape = [None, 100, 100, 3]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(64, hidden_dim, kernel_size=(9, 9), stride=(1, 1))
        self.pool = nn.AdaptiveMaxPool2d((1, 1, hidden_dim))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTMCell(hidden_dim + 4, hidden_dim)

    def forward(self, obs, cat_tensor):
        obs, (hx, cx) = obs
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
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


class ActionNetwork(nn.Module):
    def __init__(self, input_dim=256, num_actions=9):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class A2C(nn.Module):
    def __init__(self, hidden_dim=256, num_actions=9):
        super(A2C, self).__init__()
        self.shared_network = SharedNetwork(hidden_dim)
        self.value_network = ValueNetwork(hidden_dim)
        self.action_policy = ActionNetwork(hidden_dim, num_actions)

    def forward(self, x, lstm_state, cat_tensor):
        x = torch.reshape(x, (-1, 1, 100, 100))
        cat_tensor = torch.reshape(cat_tensor, (-1, 4))
        obs = (x, lstm_state)
        features, cx = self.shared_network(obs, cat_tensor)
        value = self.value_network(features)
        action = self.action_policy(features)
        return action, value, (features, cx)


class A2CGym(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim):
        super(A2CGym, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.value_network = nn.Linear(128, 1)
        self.action_policy = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.reshape(x, (-1, 4))
        x = F.relu(self.fc1(x))
        value = self.value_network(x)
        action = self.action_policy(x)
        return action, value
