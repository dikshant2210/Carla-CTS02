"""
Author: Dikshant Gupta
Time: 27.09.22 20:57
"""

import pygame
import subprocess
import time
import os
import argparse
import numpy as np
from multiprocessing import Process
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from benchmark.rl.a2c.model import A2C
from config import Config
from benchmark.environment import GIDASBenchmark
from benchmark.rl.a2c.a2ccadrl import A2CCadrl


def train_a2c(args):
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/a2c/{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

    # Path to save model
    path = "_out/a2c/"
    if not os.path.exists(path):
        os.mkdir(path)

    # Setting up environment
    env = GIDASBenchmark()
    agent = A2CCadrl(env.world, env.map, env.scene)
    env.reset_agent(agent)

    # Instantiating RL agent
    torch.manual_seed(100)
    rl_agent = A2C(hidden_dim=256, num_actions=3).cuda()
    optimizer = torch.optim.Adam(rl_agent.parameters(), lr=Config.a2c_lr)
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    load_path = args.checkpoint
    if load_path:
        current_episode = int(load_path.strip().split('/')[2].split('_')[3].split('.')[0])
        rl_agent.load_state_dict(torch.load(load_path))

    max_episodes = Config.train_episodes
    print("Total training episodes: {}".format(max_episodes))
    file.write("Total training episodes: {}\n".format(max_episodes))
    while current_episode < max_episodes:
        # Get the scenario id, parameters and instantiate the world
        total_episode_reward = 0
        observation = env.reset()

        # Setup initial inputs for LSTM Cell
        cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

        # Setup placeholders for training value logs
        values = []
        log_probs = []
        rewards = []
        entropies = []

        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
        nearmiss = False
        acccident = False

        for step_num in range(Config.num_steps):
            if Config.display:
                env.render()
            # Forward pass of the RL Agent
            # if step_num > 0:
            #     plt.imsave('_out/{}.png'.format(step_num), observation)
            input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6,
                                                    speed_action])).cuda().type(torch.cuda.FloatTensor)
            logit, value, (hx, cx) = rl_agent(input_tensor, (hx, cx), cat_tensor)

            prob = F.softmax(logit, dim=-1)
            m = Categorical(prob)
            action = m.sample()
            speed_action = action.item()

            observation, reward, done, info = env.step(speed_action)
            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or nearmiss
            acccident_current = info['accident']
            acccident = acccident_current or acccident
            total_episode_reward += reward

            # Logging value for loss calculation and backprop training
            log_prob = m.log_prob(action)
            entropy = -(F.log_softmax(logit, dim=-1) * prob).sum()
            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            if done or acccident:
                break

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(info['goal'], acccident, nearmiss))
        file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(info['goal'], acccident, nearmiss))

        ##############################################################
        # Update weights of the model
        R = 0
        rewards.reverse()
        values.reverse()
        log_probs.reverse()
        entropies.reverse()
        returns = []
        for r in rewards:
            R = Config.a2c_gamma * R + r
            returns.append(R)
        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.cuda().type(torch.cuda.FloatTensor)

        policy_losses = []
        value_losses = []

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]]).cuda()))
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() - \
               Config.a2c_entropy_coef * torch.stack(entropies).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}, Reward: {:.4f}".format(
            torch.stack(policy_losses).sum().item(), torch.stack(value_losses).sum().item(),
            torch.stack(entropies).sum(), total_episode_reward))
        file.write("Policy Loss: {:.4f}, Value Loss: {:.4f}, Reward: {:.4f}\n".format(
            torch.stack(policy_losses).sum().item(), torch.stack(value_losses).sum().item(), total_episode_reward))
        current_episode += 1
        if current_episode % Config.save_freq == 0:
            torch.save(rl_agent.state_dict(), "{}a2c_entropy_005_{}.pth".format(path, current_episode))

    env.close()
    print("Training time: {:.4f}hrs".format((time.time() - t0) / 3600))
    file.write("Training time: {:.4f}hrs\n".format((time.time() - t0) / 3600))
    torch.save(rl_agent.state_dict(), "{}a2c_entropy_{}.pth".format(path, current_episode))
    file.close()


def main(args):
    print(__doc__)

    try:
        train_a2c(args)

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
        default=2500,
        type=int,
        help='TCP port to listen to (default: 2500)')
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
