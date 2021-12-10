"""
Author: Dikshant Gupta
Time: 10.12.21 03:18
"""

import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from Rainbow.common.replay_buffer import ReplayBuffer
from Rainbow.model import RainbowCnnDQN, USE_CUDA


class Rainbow:
    def __init__(self, env, num_atoms=51, Vmin=-10, Vmax=10):
        self.env = env
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.num_atoms = 51
        self.current_model = RainbowCnnDQN(self.env.observation_space.shape, self.env.action_space.n,
                                           num_atoms, Vmin, Vmax)
        self.target_model = RainbowCnnDQN(self.env.observation_space.shape, self.env.action_space.n,
                                          num_atoms, Vmin, Vmax)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.update_target()

        self.replay_initial = 30000
        self.replay_buffer = ReplayBuffer(100000)

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        next_dist = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        with torch.no_grad():
            next_state = torch.FloatTensor(np.float32(next_state))
        if USE_CUDA:
            state = state.cuda()
            next_state = next_state.cuda()
            action = torch.LongTensor(action).cuda()
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        proj_dist = self.projection_distribution(next_state, reward, done)
        if USE_CUDA:
            proj_dist = proj_dist.cuda()

        dist = self.current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(proj_dist * dist.log()).sum(1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_model.reset_noise()
        self.target_model.reset_noise()

        return loss

    @staticmethod
    def plot(frame_idx, rewards, losses):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()

    def train(self, batch_size=32):
        num_frames = int(1e7)
        gamma = 0.99

        losses = []
        all_rewards = []
        episode_reward = 0
        num_steps = 0
        episodes = 0

        nearmiss = False
        action_count = {0: 0, 1: 0, 2: 0}

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            action = self.current_model.act(state)

            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            num_steps += 1
            nearmiss = nearmiss or info['near miss']
            action_count[action] += 1

            if done or num_steps >= 300:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episodes += 1
                print("-------------------------")
                print("Ep: {}, Sce: {}, Dist: {:.1f}, Speed: {:.1f}".format(
                    episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
                print("Goal: {}, Accident: {}, Nearmiss: {}".format(info['goal'], info['accident'], nearmiss))
                print("Total Steps: {}, Ep.Steps: {}, Ep. Reward: {:.4f}".format(frame_idx, num_steps, episode_reward))
                print(action_count)
                episode_reward = 0
                num_steps = 0
                nearmiss = False
                action_count = {0: 0, 1: 0, 2: 0}

            if len(self.replay_buffer) > self.replay_initial:
                loss = self.compute_td_loss(batch_size)
                losses.append(loss.item())

            # if frame_idx % 100 == 0:
            #     self.plot(frame_idx, all_rewards, losses)

            if frame_idx % 1000 == 0:
                self.update_target()
