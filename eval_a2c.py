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
import pickle as pkl
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.rl.a2c.model import A2C
from config import Config
from environment import GIDASBenchmark


def eval_a2c():
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/{}/{}.pkl".format("a2c", datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)

    # Path to save model
    path = "_out/a2c/"
    if not os.path.exists(path):
        os.mkdir(path)

    # Path to load model
    path = "_out/a2c/a2c_entropy_005_3000.pth"
    if not os.path.exists(path):
        print("Path: {} does not exist".format(path))

    # Setting up environment in eval mode
    env = GIDASBenchmark()
    env.reset_agent('cadrl')
    env.eval()

    # Instantiating RL agent
    # torch.manual_seed(100)
    rl_agent = A2C(hidden_dim=256, num_actions=3).cuda()
    rl_agent.load_state_dict(torch.load(path))
    rl_agent.eval()
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    max_episodes = len(env.episodes)
    print("Total eval episodes: {}".format(max_episodes))
    data_log = {}

    max_episodes = 2
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
        prev_action = 1

        episode_log = {}
        action_count = {0: 0, 1: 0, 2: 0}
        exec_time = []
        actions_list = []
        risk = []
        ped_obs = []
        impact_speed = []
        trajectory = []

        for step_num in range(Config.num_steps):
            trajectory.append((env.world.player.get_location().x,
                               env.world.player.get_location().y))
            if Config.display:
                env.render()
            # Forward pass of the RL Agent
            start_time = time.time()
            input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6,
                                                    speed_action])).cuda().type(torch.cuda.FloatTensor)
            logit, value, (hx, cx) = rl_agent(input_tensor, (hx, cx), cat_tensor)

            # print(logit.size(), torch.argmax(logit, dim=1)[0].item(), logit)
            speed_action = torch.argmax(logit, dim=1)[0].item()

            if speed_action != 0:
                total_acc_decc += 1

            observation, reward, done, info = env.step(speed_action)
            time_taken = time.time() - start_time
            exec_time.append(time_taken)
            actions_list.append(speed_action)
            if speed_action != 1 and prev_action != speed_action:
                total_acc_decc += 1
            prev_action = speed_action
            action_count[speed_action] += 1

            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or (nearmiss and speed > 0)
            accident_current = info['accident']
            accident = accident_current or (accident and speed > 0)
            total_episode_reward += reward
            speed = np.sqrt(info['velocity'].x ** 2 + info['velocity'].y ** 2)
            impact_speed.append(speed)
            risk.append(info['risk'])
            ped_obs.append(info['ped_observable'])

            if done or accident:
                break

        time_to_goal = (step_num + 1) * Config.simulation_step
        episode_log['ttg'] = time_to_goal
        episode_log['risk'] = risk
        episode_log['actions'] = actions_list
        episode_log['exec'] = exec_time
        episode_log['impact_speed'] = impact_speed
        episode_log['trajectory'] = trajectory
        episode_log['ped_dist'] = info['ped_distance']
        episode_log['scenario'] = info['scenario']
        episode_log['ped_speed'] = info['ped_speed']
        episode_log['crash'] = info['accident']
        episode_log['nearmiss'] = nearmiss
        episode_log['goal'] = info['goal']
        episode_log['ped_observable'] = ped_obs
        data_log[current_episode + 1] = episode_log

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}'.format(
            info['goal'], info['accident'], nearmiss, total_episode_reward))
        print('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms, Action: {}'.format(
            time_to_goal, total_acc_decc, sum(exec_time) * 1000 / len(exec_time), action_count))
        ##############################################################

        current_episode += 1

    env.close()
    with open(filename, "wb") as write_file:
        pkl.dump(data_log, write_file)
    print("Log file written here: {}".format(filename))
    print('-' * 60)


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
        default=2900,
        type=int,
        help='TCP port to listen to (default: 2900)')
    arg_parser.add_argument('--test', type=str, default='')
    arg = arg_parser.parse_args()
    Config.port = arg.port
    if arg.test:
        Config.test_scenarios = [arg.test]

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main()
