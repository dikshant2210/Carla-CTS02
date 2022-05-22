"""
Author: Dikshant Gupta
Time: 10.11.21 01:07
"""

import pygame
import subprocess
import argparse
import time
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
    filename = "_out/isdespot/{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

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
    file.write("Total training episodes: {}\n".format(max_episodes))
    pedestrian_path = {}
    while current_episode < max_episodes:
        # Get the scenario id, parameters and instantiate the world
        total_episode_reward = 0
        observation = env.reset()
        nearmiss = False
        accident = False
        exec_time = 0
        count = 0
        total_acc_decc = 0
        ped_data = []
        step_num = 0
        prev_action = 1
        risk = 0

        for step_num in range(Config.num_steps):
            ped_data.append((env.world.walker.get_location().x, env.world.walker.get_location().y))
            if Config.display:
                env.render()

            start_time = time.time()
            observation, reward, done, info = env.step(action=None)
            if env.planner_agent.pedestrian_observable:
                exec_time += (time.time() - start_time)
                count += 1

            if env.control.throttle != 0:
                action = 0
            elif env.control.brake != 0:
                action = 2
            else:
                action = 1
            if action != 1 and prev_action != action:
                total_acc_decc += 1
            prev_action = action

            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or (nearmiss and speed > 0)
            accident_current = info['accident']
            accident = accident_current or (accident and speed > 0)
            total_episode_reward += reward
            risk += info['risk']

            if done or accident:
                break
        current_episode += 1
        pedestrian_path[current_episode] = ped_data

        # Evaluate episode statistics(Crash rate, nearmiss rate, time to goal, smoothness, execution time, violations)
        time_to_goal = (step_num + 1) * Config.simulation_step
        if count == 0:
            exec_time = 0
        else:
            exec_time = exec_time / count

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode, info['scenario'], info['ped_speed'], info['ped_distance']))
        file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
            current_episode, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}, Reward: {:.4f}, Risk: {:.4f}'.format(
            info['goal'], accident, nearmiss, total_episode_reward, risk / count))
        file.write('Goal reached: {}, Accident: {}, Nearmiss: {}, Risk: {:.4f}\n'.format(info['goal'], accident,
                                                                                         nearmiss, risk / count))
        print('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms'.format(
            time_to_goal, total_acc_decc, exec_time * 1000))
        file.write('Time to goal: {:.4f}s, #Acc/Dec: {}, Execution time: {:.4f}ms\n'.format(
            time_to_goal, total_acc_decc, exec_time * 1000))

        ##############################################################

    env.close()
    print("Testing time: {:.4f}hrs".format((time.time() - t0) / 3600))
    file.write("Testing time: {:.4f}hrs\n".format((time.time() - t0) / 3600))
    file.close()
    with open('_out/pedestrian_data.pkl', 'wb') as file:
        pkl.dump(pedestrian_path, file)


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
