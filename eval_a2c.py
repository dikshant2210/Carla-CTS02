"""
Author: Dikshant Gupta
Time: 05.10.21 10:47
"""

import pygame
import argparse
import subprocess
import time
import os
import numpy as np
from multiprocessing import Process
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.rl.a2c.model import A2C
from config import Config
from environment import GIDASBenchmark


def eval_a2c():
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/a2c/test_{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

    # Path to save model
    path = "_out/a2c/"
    if not os.path.exists(path):
        os.mkdir(path)

    # Path to load model
    path = "_out/a2c/a2c_entropy_005_13000.pth"
    if not os.path.exists(path):
        print("Path: {} does not exist".format(path))

    # Setting up environment in eval mode
    env = GIDASBenchmark()
    env.eval()

    # Instantiating RL agent
    torch.manual_seed(100)
    rl_agent = A2C(hidden_dim=256, num_actions=3).cuda()
    rl_agent.load_state_dict(torch.load(path))
    rl_agent.eval()
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    max_episodes = len(env.episodes)
    print("Total eval episodes: {}".format(max_episodes))
    file.write("Total eval episodes: {}\n".format(max_episodes))
    while current_episode < max_episodes:
        # Get the scenario id, parameters and instantiate the world
        total_episode_reward = 0
        observation = env.reset()

        # Setup initial inputs for LSTM Cell
        cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

        # Setup placeholders
        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
        nearmiss = False
        accident = False
        step_num = 0

        total_acc_decc = 0
        exec_time = 0

        for step_num in range(Config.num_steps):
            if Config.display or True:
                env.render()
            # Forward pass of the RL Agent
            start_time = time.time()
            input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6,
                                                    speed_action])).cuda().type(torch.cuda.FloatTensor)
            logit, value, (hx, cx) = rl_agent(input_tensor, (hx, cx), cat_tensor)

            prob = F.softmax(logit, dim=-1)
            m = Categorical(prob)
            action = m.sample()
            speed_action = action.item()

            if speed_action != 0:
                total_acc_decc += 1

            observation, reward, done, info = env.step(speed_action)
            exec_time += (time.time() - start_time)

            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or nearmiss
            accident_current = info['accident']
            accident = accident_current or accident
            total_episode_reward += reward

            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y

            if done or accident:
                break

        # Evaluate episode statistics(Crash rate, nearmiss rate, time to goal, smoothness, execution time, violations)
        time_to_goal = (step_num + 1) * Config.simulation_step
        exec_time = exec_time / (step_num + 1)
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(info['goal'], accident, nearmiss))
        file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(info['goal'], accident, nearmiss))
        print('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms'.format(
            time_to_goal, total_acc_decc, exec_time))
        file.write('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms\n'.format(
            time_to_goal, total_acc_decc, exec_time * 1000))

        ##############################################################

        current_episode += 1
        if current_episode % Config.save_freq == 0:
            torch.save(rl_agent.state_dict(), "{}a2c_{}.pth".format(path, current_episode))

    env.close()
    print("Evaluation time: {:.4f}hrs".format((time.time() - t0) / 3600))
    file.write("Evaluation time: {:.4f}hrs\n".format((time.time() - t0) / 3600))
    torch.save(rl_agent.state_dict(), "{}a2c_{}.pth".format(path, current_episode))
    file.close()


def main():
    print(__doc__)

    try:
        eval_a2c()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    arg_parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2900)')
    arg = arg_parser.parse_args()
    Config.port = arg.port

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main()
