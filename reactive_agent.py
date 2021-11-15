"""
Author: Dikshant Gupta
Time: 15.11.21 16:29
"""

import pygame
import subprocess
import time
from multiprocessing import Process
from datetime import datetime

from config import Config
from environment import GIDASBenchmark


def reactive_controller():
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/reactive/{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

    # Setting up environment
    env = GIDASBenchmark()
    env.reset_agent('reactive')
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    max_episodes = len(env.episodes)
    nearmiss = False
    print("Total training episodes: {}".format(max_episodes))
    file.write("Total training episodes: {}\n".format(max_episodes))
    while current_episode < max_episodes:
        # Get the scenario id, parameters and instantiate the world
        total_episode_reward = 0
        observation = env.reset()

        for step_num in range(Config.num_steps):
            if Config.display:
                env.render()

            observation, reward, done, info = env.step(action=None)
            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or nearmiss
            total_episode_reward += reward

            if done:
                break
        current_episode += 1

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(info['goal'], info['accident'], nearmiss))
        file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(info['goal'], info['accident'], nearmiss))

        ##############################################################

    env.close()
    print("Testing time: {:.4f}hrs".format((time.time() - t0) / 3600))
    file.write("Testing time: {:.4f}hrs\n".format((time.time() - t0) / 3600))
    file.close()


def main():
    print(__doc__)

    try:
        reactive_controller()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)


if __name__ == '__main__':
    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main()

