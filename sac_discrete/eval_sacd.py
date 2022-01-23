"""
Author: Dikshant Gupta
Time: 13.12.21 11:31
"""

import numpy as np
import torch
from datetime import datetime
import time

from .base import BaseAgent
from config import Config
from sac_discrete.sacd.model import DQNBase, TwinnedQNetwork, CateoricalPolicy


class EvalSacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000, save_interval=100000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0, current_episode=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps, save_interval,
            log_interval, eval_interval, cuda, seed)

        # Define networks.
        self.conv = DQNBase(
            self.env.observation_space.shape[2]).to(self.device)
        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n,
            shared=True).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=dueling_net, shared=True).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=dueling_net, shared=True).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        path = "_out/GIDASBenchmark/summary/best/"
        self.conv.load_state_dict(torch.load(path + "conv.pth"))
        self.policy.load_state_dict(torch.load(path + "policy.pth"))
        self.conv.eval()
        self.policy.eval()

        filename = "_out/hylear/{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
        print(filename)
        self.file = open(filename, "w")
        self.file.write(str(vars(Config)) + "\n")
        self.current_episode = current_episode

    def evaluate(self):
        num_episodes = self.current_episode
        total_episodes = len(self.env.episodes) + self.current_episode
        print("Total testing episodes: {}".format(total_episodes))
        self.file.write("Total testing episodes: {}\n".format(total_episodes))
        num_steps = 0
        total_return = 0.0
        print('-' * 60)

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            exec_time = 0
            done = False
            nearmiss = False
            action_count = {0: 0, 1: 0, 2: 0}

            prev_action = 1
            total_acc_decc = 0

            t = np.zeros(6)  # reward, vx, vt, onehot last action
            t[3 + 1] = 1.0  # index = 3 + last_action(maintain)

            while (not done) and episode_steps < self.max_episode_steps:
                # self.test_env.render()

                start_time = time.time()
                action = self.exploit((state, t))
                next_state, reward, done, info = self.test_env.step(action)
                exec_time += (time.time() - start_time)
                if action != 1 and prev_action != action:
                    total_acc_decc += 1
                prev_action = action

                done = done or info["accident"]
                action_count[action] += 1
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state
                t = np.zeros(6)
                t[0] = max(min(reward, 2.0), -2.0)
                t[1] = info['velocity'].x / Config.max_speed
                t[2] = info['velocity'].y / Config.max_speed
                t[3 + action] = 1.0
                nearmiss = nearmiss or info["near miss"]

            num_episodes += 1
            total_return += episode_return
            exec_time = exec_time / episode_steps
            time_to_goal = episode_steps * Config.simulation_step
            print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
                num_episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
            self.file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
                num_episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
            print('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}'.format(
                info['goal'], info['accident'], nearmiss, episode_return))
            self.file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(
                info['goal'], info['accident'], nearmiss))
            print('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms, Action: {}'.format(
                time_to_goal, total_acc_decc, exec_time * 1000, action_count))
            self.file.write('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms\n'.format(
                time_to_goal, total_acc_decc, exec_time * 1000))

            if num_episodes >= total_episodes:
                break

        print('-' * 60)
        self.file.close()

    def exploit(self, state):
        # Act without randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action = self.policy.act(state)
        return action.item()

    def explore(self, state):
        pass

    def update_target(self):
        pass

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    def calc_critic_loss(self, batch, weights):
        pass

    def calc_policy_loss(self, batch, weights):
        pass

    def calc_entropy_loss(self, entropies, weights):
        pass

    def save_models(self, save_dir):
        pass

    def __del__(self):
        self.file.close()
