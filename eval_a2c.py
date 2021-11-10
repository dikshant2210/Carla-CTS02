"""
Author: Dikshant Gupta
Time: 12.10.21 13:33
"""


import carla
import pygame
import subprocess
import time
import os
import random
import numpy as np
from multiprocessing import Process
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from world import World
from hud import HUD
from agents.rl.a2c.model import A2C
from agents.navigation.rlagent import RLAgent
from config import Config
from agents.tools.scenario import Scenario


class Environment:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.client = carla.Client(Config.host, Config.port)
        self.client.set_timeout(120.0)

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
        self.planner_agent = RLAgent(self.world, self.map, self.scene)

        wld_map = wld.get_map()
        print(wld_map.name)

    def get_observation(self):
        control, observation = self.planner_agent.run_step()
        return control, observation

    def step(self, action):
        self.world.player.apply_control(action)
        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        _, observation = self.planner_agent.run_step()
        reward, goal, accident, near_miss = self.planner_agent.get_reward()

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
    filename = "_out/a2c/test_{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

    # Path to load model
    path = "_out/a2c/seg_inp_all_scenarios/a2c_3000.pth"
    if not os.path.exists(path):
        print("Path: {} does not exist".format(path))

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

    # Instantiating RL agent
    torch.manual_seed(100)
    rl_agent = A2C(hidden_dim=256, num_actions=3).cuda()
    rl_agent.load_state_dict(torch.load(path))
    rl_agent.eval()
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
        env.reset(scenario_id, ped_speed, ped_distance)
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, scenario_id, ped_speed, ped_distance))
        file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
            current_episode + 1, scenario_id, ped_speed, ped_distance))

        # Setup initial inputs for LSTM Cell
        cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
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
            ##############################################################

            ##############################################################
            # Forward pass of the RL Agent
            input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x, velocity_y, speed_action])).cuda().type(
                torch.cuda.FloatTensor)
            logit, value, (hx, cx) = rl_agent(input_tensor, (hx, cx), cat_tensor)

            prob = F.softmax(logit, dim=-1)
            m = Categorical(prob)
            action = m.sample()
            speed_action = action.item()
            if speed_action == 2:
                print(speed_action)
            if speed_action == 0:
                control.throttle = 0.6
            elif speed_action == 2:
                control.throttle = -0.6
            else:
                control.throttle = 0
            ##############################################################

            env.world.tick(clock)
            if Config.display:
                env.world.render(env.display)
                pygame.display.flip()
            _, reward, goal, accident, nearmiss_current = env.step(control)
            done = goal or accident
            nearmiss = nearmiss_current or nearmiss

            ##############################################################

            velocity = env.planner_agent.vehicle.get_velocity()
            velocity_x = velocity.x
            velocity_y = velocity.y

            if done:
                break
            ##############################################################
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, accident, nearmiss))
        file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(goal, accident, nearmiss))
        current_episode += 1

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
    # subprocess.run(['cd /opt/carla-simulator && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)
    # subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True)


if __name__ == '__main__':
    # p = Process(target=run_server)
    # p.start()
    # time.sleep(5)  # wait for the server to start

    main()
    # p.terminate()

