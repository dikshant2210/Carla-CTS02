"""
Author: Dikshant Gupta
Time: 10.11.21 01:07
"""

import pygame
import subprocess
import argparse
import time
import json
import pickle as pkl
import numpy as np
from multiprocessing import Process
from datetime import datetime

from config import Config
from environment import GIDASBenchmark


def eval_isdespot(arg):
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/isdespot/{}.pkl".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)

    # Setting up environment
    print("Environment port: {}".format(Config.port))
    env = GIDASBenchmark(port=Config.port)
    env.reset_agent(arg.agent)
    env.eval(arg.episode)
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    max_episodes = len(env.episodes)
    print("Total testing episodes: {}".format(max_episodes))
    pedestrian_path = {}
    data_log = {}
    while current_episode < max_episodes:
        # Get the scenario id, parameters and instantiate the world
        total_episode_reward = 0
        observation = env.reset()
        nearmiss = False
        accident = False
        count = 0
        ped_data = []
        step_num = 0

        episode_log = {}
        exec_time = []
        actions_list = []
        risk = []
        ped_obs = []
        impact_speed = []
        trajectory = []

        for step_num in range(Config.num_steps):
            ped_data.append((env.world.walker.get_location().x, env.world.walker.get_location().y))
            trajectory.append((env.world.player.get_location().x, env.world.player.get_location().y))
            if Config.display:
                env.render()

            start_time = time.time()
            observation, reward, done, info = env.step(action=None)
            if env.planner_agent.pedestrian_observable:
                time_taken = (time.time() - start_time)
                count += 1
            else:
                time_taken = 0
            exec_time.append(time_taken)

            if env.control.throttle != 0:
                action = 0
            elif env.control.brake != 0:
                action = 2
            else:
                action = 1
            actions_list.append(action)

            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
            impact_speed.append(speed)

            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or (nearmiss and speed > 0)
            accident_current = info['accident']
            accident = accident_current or (accident and speed > 0)
            total_episode_reward += reward
            risk.append(info['risk'])
            ped_obs.append(info['ped_observable'])

            if done or accident:
                break
        current_episode += 1
        pedestrian_path[current_episode] = ped_data

        # Evaluate episode statistics(Crash rate, nearmiss rate, time to goal, smoothness, execution time, violations)
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
        episode_log['crash'] = accident
        episode_log['nearmiss'] = nearmiss
        episode_log['ped_observable'] = ped_obs
        data_log[current_episode] = episode_log

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}, Risk: {:.4f}'.format(
            info['goal'], accident, nearmiss, total_episode_reward, sum(risk) / step_num))
        print('Time to goal: {:.4f}s, Execution time: {:.4f}ms'.format(
            time_to_goal, sum(exec_time) * 1000 / (count + 1)))
        ##############################################################

    env.close()
    print("Testing time: {:.4f}hrs".format((time.time() - t0) / 3600))
    with open(filename, "wb") as write_file:
        pkl.dump(data_log, write_file)
    with open('_out/pedestrian_data.pkl', 'wb') as file:
        pkl.dump(pedestrian_path, file)
    print("log file written!!")


def main(arg):
    print(__doc__)

    try:
        eval_isdespot(arg)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    port = "-carla-port={}".format(Config.port)
    print("Server port: {}".format(Config.port))
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl ' + port], shell=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    arg_parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2300)')
    arg_parser.add_argument(
        '-ep', '--episode',
        default=0,
        type=int,
        help='episode number to resume from')
    arg_parser.add_argument('--test', type=str, default='')
    arg_parser.add_argument('--agent', type=str, default='isdespot')
    arg_parser.add_argument('--despot_port', type=int, default=1255)
    args = arg_parser.parse_args()
    Config.port = args.port
    Config.despot_port = args.despot_port
    if args.test:
        Config.test_scenarios = [args.test]

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main(args)
