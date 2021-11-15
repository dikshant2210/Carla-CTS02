"""
Author: Dikshant Gupta
Time: 10.11.21 01:07
"""

import carla
import pygame
import subprocess
import time
import random
import numpy as np
from multiprocessing import Process
from datetime import datetime


from environment.world import World
from environment.hud import HUD
from agents.tools.connector import Connector
from agents.navigation.isdespot import ISDespotP
from config import Config
from agents.tools.scenario import Scenario


class Environment:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.client = carla.Client(Config.host, Config.port)
        self.client.set_timeout(60.0)

        self.display = None
        if Config.display:
            self.display = pygame.display.set_mode(
                (Config.width, Config.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0, 0, 0))
            pygame.display.flip()

        hud = HUD(Config.width, Config.height)
        self.client.load_world('Town01_Opt')
        wld = self.client.get_world()
        wld.unload_map_layer(carla.MapLayer.StreetLights)
        wld.unload_map_layer(carla.MapLayer.Props)
        self.map = wld.get_map()
        settings = wld.get_settings()
        settings.fixed_delta_seconds = Config.simulation_step
        settings.synchronous_mode = Config.synchronous
        wld.apply_settings(settings)

        self.scene_generator = Scenario(wld)
        self.scene = self.scene_generator.scenario01()
        self.world = World(wld, hud, self.scene, Config)

        conn = Connector(Config.despot_port)
        self.planner_agent = ISDespotP(self.world, self.map, self.scene, conn)

        wld_map = wld.get_map()
        print(wld_map.name)

    def get_observation(self):
        control, observation = self.planner_agent.run_step()
        return control, observation

    def step(self, action):
        self.world.player.apply_control(action)
        if action.throttle == 0.6:
            speed_action = 0
        elif action.brake == 0.6:
            speed_action = 2
        else:
            speed_action = 1
        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        _, observation = self.planner_agent.run_step()
        reward, goal, accident, near_miss = self.planner_agent.get_reward(speed_action)

        return observation, reward, goal, accident, near_miss

    def reset(self, scenario_id, ped_speed, ped_distance):
        func = 'self.scene_generator.scenario' + scenario_id
        scenario = eval(func + '()')
        self.world.restart(scenario, ped_speed, ped_distance)
        self.planner_agent.update_scenario(scenario)


def eval_a2c():
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/a2c/despot_{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

    # Compiling all different combination of scenarios
    episodes = list()
    for scenario in Config.test_scenarios:
        for speed in np.arange(Config.test_ped_speed_range[0], Config.test_ped_speed_range[1] + 1, 0.1):
            for distance in np.arange(Config.test_ped_distance_range[0], Config.test_ped_distance_range[1] + 1, 1):
                episodes.append((scenario, speed, distance))
    random.shuffle(episodes)
    ##############################################################

    ##############################################################
    # Setting up environment
    env = Environment()
    ##############################################################

    ##############################################################
    # Simulation loop
    current_episode = 0
    max_episodes = len(episodes)
    print("Total training episodes: {}".format(max_episodes))
    file.write("Total training episodes: {}\n".format(max_episodes))
    while current_episode < max_episodes:
        ##############################################################
        # Get the scenario id, parameters and instantiate the world
        idx = current_episode % len(episodes)
        scenario_id, ped_speed, ped_distance = episodes[idx]
        ped_speed = 1.8
        ped_distance = 35.0
        env.reset(scenario_id, ped_speed, ped_distance)
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, scenario_id, ped_speed, ped_distance))
        file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
            current_episode + 1, scenario_id, ped_speed, ped_distance))

        goal = False
        accident = False
        nearmiss = False
        ##############################################################

        clock = pygame.time.Clock()
        env.client.get_world().tick()
        for _ in range(Config.num_steps):
            clock.tick_busy_loop(60)

            ##############################################################
            # Get the current observation
            control, observation = env.get_observation()

            env.world.tick(clock)
            if Config.display:
                env.world.render(env.display)
                pygame.display.flip()
            _, reward, goal, accident, nearmiss_current = env.step(control)
            done = goal or accident
            nearmiss = nearmiss_current or nearmiss

            ##############################################################

            if done:
                break
            ##############################################################
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, accident, nearmiss))
        file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(goal, accident, nearmiss))
        current_episode += 1

    print("Evaluation time: {:.4f}hrs".format((time.time() - t0) / 3600))
    file.write("Evaluation time: {:.4f}hrs\n".format((time.time() - t0) / 3600))
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
    # subprocess.run(['cd /opt/carla-simulator && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)
    # subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True)


if __name__ == '__main__':
    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main()
    # p.terminate()
