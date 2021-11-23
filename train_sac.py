"""
Author: Dikshant Gupta
Time: 24.10.21 00:47
"""

import subprocess
from multiprocessing import Process
import numpy as np
import os
import pygame
import argparse
import time
import torch
from datetime import datetime
from torch.optim import Adam
import torch.nn.functional as F

from config import Config
from environment.environment import GIDASBenchmark
from agents.rl.sac.model import SAC, QNetwork
from agents.rl.sac.utils import soft_update, hard_update, ExpBuffer


class SACTrainer:
    def __init__(self):
        ##############################################################
        # Logging file
        filename = "_out/sac/{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
        print(filename)
        self.file = open(filename, "w")
        self.file.write(str(vars(Config)) + "\n")

        # Path to save model
        path = "_out/sac/"
        if not os.path.exists(path):
            os.mkdir(path)

        # Initialize environment and experience buffer
        replay_buffer_size = Config.batch_size * Config.num_steps
        sample_length = 300
        self.exp_buffer = ExpBuffer(replay_buffer_size, sample_length)
        self.env = GIDASBenchmark()

        # Instantiating the RL Agent
        torch.manual_seed(100)
        self.rl_agent = SAC(Config.num_actions).cuda()
        self.critic_optim = Adam(list(self.rl_agent.q_network.parameters()) +
                                 list(self.rl_agent.shared_network.parameters()), lr=Config.sac_lr)
        self.critic_target = QNetwork(Config.num_actions).cuda()
        hard_update(self.critic_target, self.rl_agent.q_network)
        self.policy_optim = Adam(list(self.rl_agent.action_policy.parameters()) +
                                 list(self.rl_agent.shared_network.parameters()), lr=Config.sac_lr)

        # Parameters
        self.gamma = Config.sac_gamma
        self.tau = Config.sac_tau
        self.alpha = Config.sac_alpha
        ##############################################################

    def train(self):
        ##############################################################
        # Simulation loop
        max_episodes = Config.train_episodes
        print("Total training episodes: {}".format(max_episodes))
        current_episode = 0
        total_steps = 0
        pre_training = False

        while current_episode < max_episodes:
            obs = self.env.reset()
            obs = torch.from_numpy(obs).cuda().type(torch.cuda.FloatTensor)
            obs = torch.reshape(obs, (-1, 3, 400, 400))

            # Setup placeholder variables
            total_episode_reward = 0
            reward = 0
            last_action = 1
            velocity_x = 0
            velocity_y = 0
            near_miss = False
            accident = False
            info = None

            # Setup initial inputs for LSTM Cell
            cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
            hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

            for step_num in range(Config.num_steps):
                if Config.display:
                    self.env.render()
                if total_steps == Config.batch_size * Config.num_steps:
                    pre_training = True
                    print("Pre-train phase done.")

                # Forward pass of the RL Agent
                cat_tensor = torch.from_numpy(np.array([reward, velocity_x, velocity_y, last_action])).cuda().type(
                    torch.cuda.FloatTensor)
                cat_tensor = torch.reshape(cat_tensor, (-1, 4))
                with torch.no_grad():
                    state = self.rl_agent.shared_network((obs, (hx, cx)), cat_tensor)
                    a, _, _ = self.rl_agent.action_policy.sample(state[0])
                a = a.cpu().numpy()
                speed_action = np.argmax(a, axis=-1)[0]
                if not pre_training:
                    speed_action = self.env.action_space.sample()
                    a = np.zeros((1, 3))
                    a[0, speed_action] = 1.0

                # Simulate one step
                next_obs, reward, done, info = self.env.step(speed_action)
                next_obs = torch.from_numpy(next_obs).cuda().type(torch.cuda.FloatTensor)
                next_obs = torch.reshape(next_obs, (-1, 3, 400, 400))
                total_episode_reward += reward

                # Add transition to the buffer
                next_cat = torch.from_numpy(np.array([reward, info['velocity'].x, info['velocity'].y, speed_action]))
                self.exp_buffer.write_tuple([obs.cpu(), hx.cpu(), cx.cpu(), a, reward, next_obs.cpu(), state[0].cpu(),
                                             state[1].cpu(), cat_tensor.cpu(), next_cat])

                # Update placeholders and state variables
                last_action = speed_action
                obs = next_obs
                velocity = info['velocity']
                velocity_x = velocity.x
                velocity_y = velocity.y
                hx = state[0]
                cx = state[1]
                goal = info['goal']
                near_miss = near_miss or info['near miss']
                accident = accident or info['accident']
                total_steps += 1

                if done:
                    break

            print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
                current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
            self.file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
                current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
            print('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}'.format(
                info['goal'], accident, near_miss, total_episode_reward))
            self.file.write('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}\n'.format(
                info['goal'], accident, near_miss, total_episode_reward))
            current_episode += 1
            if pre_training and current_episode > Config.batch_size:
                self.update_parameters()

    def update_parameters(self):
        obs, hx, cx, action, rewards, next_obs, next_hx, next_cx, cat, next_cat = self.exp_buffer.sample(
            Config.batch_size)
        with torch.no_grad():
            features, _ = self.rl_agent.shared_network((next_obs, (next_hx, next_cx)), next_cat)
            next_state_action, next_state_log_pi, _ = self.rl_agent.action_policy.sample(features)
            qf1_next_target, qf2_next_target = self.critic_target(features, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        f, _ = self.rl_agent.shared_network((obs, (hx, cx)), cat)
        qf1, qf2 = self.rl_agent.q_network(f, action)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        pi, log_pi, _ = self.rl_agent.action_policy.sample(f)
        qf1_pi, qf2_pi = self.rl_agent.q_network(f, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.critic_optim.zero_grad()
        self.policy_optim.zero_grad()
        loss = qf_loss + policy_loss
        loss.backward()
        self.critic_optim.step()
        self.policy_optim.step()

        soft_update(self.critic_target, self.rl_agent.q_network, self.tau)
        print("Q-loss: {:.4f}, Policy loss: {:.4f}".format(qf_loss.detach().cpu(), policy_loss.detach().cpu()))


def main(args):
    print(__doc__)
    trainer = SACTrainer()

    try:
        trainer.train()

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
        help='TCP port to listen to (default: 2000)')
    arg_parser.add_argument(
        '-ckp', '--checkpoint',
        default='',
        type=str,
        help='load the model from this checkpoint')
    arg = arg_parser.parse_args()
    Config.port = arg.port

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main(arg)
    p.terminate()
