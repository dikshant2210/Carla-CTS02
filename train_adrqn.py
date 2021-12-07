"""
Author: Dikshant Gupta
Time: 29.11.21 22:14
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from collections import deque
import random
import math
import pygame
from multiprocessing import Process
import subprocess
import time
import argparse

from config import Config
from environment import GIDASBenchmark


class QNetwork(nn.Module):
    def __init__(self, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_actions = num_actions
        # input_shape = [None, 400, 400, 3]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(9, 9), stride=(3, 3))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(9, 9), stride=(1, 1))
        self.conv6 = nn.Conv2d(128, hidden_dim, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(hidden_dim + 4, hidden_dim, batch_first=True)
        self.qvalues = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs, lens, hidden=None):
        obs, cat_tensor = obs
        batch_size, seq_len, h, w, c = obs.size()
        obs = obs.view(batch_size * seq_len, h, w, c).permute(0, 3, 1, 2)
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, self.hidden_dim)
        x = torch.cat((x, cat_tensor), dim=-1)
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        if hidden is not None:
            x, h = self.lstm(x, hidden)
        else:
            x, h = self.lstm(x)
        x, output_lengths = pad_packed_sequence(x, batch_first=True)
        x = self.qvalues(x)
        return x, h

    def act(self, observation, lens, epsilon, hidden=None):
        out, h = self.forward(observation, lens, hidden)
        if np.random.uniform() > epsilon:
            a = torch.argmax(out).item()
        else:
            a = np.random.randint(self.n_actions)
        return a, h


class EpisodeMemory:
    def __init__(self):
        self.state = []
        self.cat_tensor = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.next_cat_tensor = []
        self.mask = []

    def add(self, state, cat_tensor, action, reward, next_state, next_cat_tensor, mask):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.mask.append(mask)
        self.cat_tensor.append(cat_tensor)
        self.next_cat_tensor.append(next_cat_tensor)

    def get_tensor_episode(self):
        state = torch.vstack([s for s in self.state])
        action = torch.vstack([torch.tensor(a) for a in self.action]).long()
        reward = torch.vstack([torch.tensor(r) for r in self.reward]).type(torch.FloatTensor)
        next_state = torch.vstack([ns for ns in self.next_state])
        mask = torch.vstack([torch.tensor(m) for m in self.mask])
        cat_tensor = torch.vstack([c for c in self.cat_tensor]).type(torch.FloatTensor)
        next_cat_tensor = torch.vstack([nc for nc in self.next_cat_tensor]).type(torch.FloatTensor)
        return state, cat_tensor, action, reward, next_state, next_cat_tensor, mask


class ADRQNTrainer:
    def __init__(self, num_actions=3):
        self.device = torch.device("cuda")
        self.lr = 0.01
        self.n_actions = num_actions
        self.gamma = 0.999
        self.explore = 30
        self.path = '_out/sac/'

        self.network = QNetwork(num_actions).to(self.device)
        self.target_network = QNetwork(num_actions).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.env = GIDASBenchmark()
        self.storage = deque([], maxlen=Config.episode_buffer)

    def train(self):
        training_episodes = Config.train_episodes
        eps_start = 0.9
        eps = eps_start
        eps_end = 0.05
        eps_decay = 10

        for i_episode in range(training_episodes):
            hidden = None
            episode_reward = 0
            state = self.env.reset()
            state = torch.tensor(state).float().unsqueeze(0)
            episode_memory = EpisodeMemory()

            a = 1
            reward = 0
            velocity_x = 0
            velocity_y = 0
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6, a]))
            cat_tensor = cat_tensor.type(torch.FloatTensor).view(1, 4)
            nearmiss = False
            acccident = False

            for t in range(Config.num_steps):
                lens = torch.IntTensor([1])
                cat_tensor = cat_tensor.type(torch.FloatTensor)
                action, hidden = self.network.act((state.unsqueeze(0).to(self.device),
                                                   cat_tensor.unsqueeze(0).to(self.device)), lens,
                                                  hidden=hidden, epsilon=eps)
                observation, reward, done, info = self.env.step(action)
                next_state = torch.tensor(observation).float().unsqueeze(0)
                velocity = info['velocity']
                velocity_x = velocity.x
                velocity_y = velocity.y
                next_cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6, a])).view(1, 4)
                episode_reward += reward
                mask = 0 if t+1 == Config.num_steps else float(done)
                nearmiss = nearmiss or info['near miss']
                acccident = acccident or info['accident']

                episode_memory.add(state, cat_tensor, action, reward, next_state, next_cat_tensor, mask)
                state = next_state
                cat_tensor = next_cat_tensor

                if done or acccident:
                    break
            self.storage.append(episode_memory)
            # Updating Networks
            if i_episode >= self.explore:
                for _ in range(Config.update_freq):
                    self.update_parameters()
                eps = eps_end + (eps_start - eps_end) * math.exp((-1 * (i_episode - self.explore)) / eps_decay)

            print("Episode: {} Ped. Speed: {:.2f}m/s, Ped. Distance: {:.2f} Eps: {:.3f}".format(
                i_episode + 1, info['ped_speed'], info['ped_distance'], eps))
            print("Goal: {}, Acccident: {}, Nearmiss: {}, Reward: {:.4f}".format(
                info['goal'], acccident, nearmiss, episode_reward))
            self.target_network.load_state_dict(self.network.state_dict())
            if (i_episode + 1) % 100 == 0:
                torch.save(self.network.state_dict(), self.path + 'sac_{}.pth'.format(i_episode + 1))
        self.env.close()

    def update_parameters(self):
        episodes = random.sample(self.storage, Config.batch_size)
        s = list()
        c = list()
        a = list()
        r = list()
        ns = list()
        nc = list()
        m = list()
        lens = list()
        for i in range(Config.batch_size):
            state_batch, cat_batch, action_batch, reward_batch, next_state_batch, next_cat_batch, mask_batch = episodes[
                i].get_tensor_episode()
            s.append(state_batch)
            c.append(cat_batch)
            a.append(action_batch)
            r.append(reward_batch)
            ns.append(next_state_batch)
            nc.append(next_cat_batch)
            m.append(mask_batch)
            lens.append(state_batch.size()[0])

        state_batch = torch.nn.utils.rnn.pad_sequence(s, batch_first=True).to(self.device)
        cat_batch = torch.nn.utils.rnn.pad_sequence(c, batch_first=True).to(self.device)
        action_batch = torch.nn.utils.rnn.pad_sequence(a, batch_first=True).to(self.device)
        reward_batch = torch.nn.utils.rnn.pad_sequence(r, batch_first=True).to(self.device)
        next_state_batch = torch.nn.utils.rnn.pad_sequence(ns, batch_first=True).to(self.device)
        next_cat_batch = torch.nn.utils.rnn.pad_sequence(nc, batch_first=True).to(self.device)
        mask_batch = torch.nn.utils.rnn.pad_sequence(m, batch_first=True).to(self.device)
        lens = torch.tensor(lens)

        # print(state_batch.size(), cat_batch.size(), action_batch.size(), reward_batch.size(), next_state_batch.size(),
        #       next_cat_batch.size(), mask_batch.size())

        q_values, _ = self.network.forward((state_batch, cat_batch), lens)
        prob = F.softmax(q_values, dim=-1)
        log_prob = F.log_softmax(q_values, dim=-1)
        entropy = -(log_prob * prob).sum()
        q_values = torch.gather(q_values, -1, action_batch).squeeze(-1)
        predicted_q_values, _ = self.target_network.forward((next_state_batch, next_cat_batch), lens)
        target_values = reward_batch.squeeze(-1) + (self.gamma * (1 - mask_batch.float().squeeze(-1)) *
                                                    torch.max(predicted_q_values, dim=-1, keepdim=False)[0])

        # Update network parameters
        self.optimizer.zero_grad()
        qloss = F.smooth_l1_loss(q_values, target_values.detach())
        loss = qloss - Config.adrqn_entropy_coef * entropy
        loss.backward()
        self.optimizer.step()
        print("Q-Loss: {:.4f}, Entropy: {:.4f}".format(qloss.item(), Config.adrqn_entropy_coef * entropy.item()))


def main(args):
    print(__doc__)

    try:
        sac_trainer = ADRQNTrainer()
        sac_trainer.train()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    port = "-carla-port={}".format(Config.port)
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl ' + port], shell=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    arg_parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2800)')
    arg_parser.add_argument(
        '-ckp', '--checkpoint',
        default='',
        type=str,
        help='load the model from this checkpoint')
    arg_parser.add_argument('--cuda', action="store_true",
                            help='run on CUDA (default: False)')
    arg = arg_parser.parse_args()
    Config.port = arg.port

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main(arg)
