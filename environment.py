"""
Author: Dikshant Gupta
Time: 11.11.21 15:48
"""

import gym
import numpy as np
import carla
import pygame
import random

from world import World
from hud import HUD
from agents.navigation.rlagent import RLAgent
from config import Config
from agents.tools.scenario import Scenario


class GIDASBenchmark(gym.Env):
    def __init__(self):
        super(GIDASBenchmark, self).__init__()

        self.action_space = gym.spaces.Discrete(Config.N_DISCRETE_ACTIONS)
        height = int(Config.segcam_image_x)
        width = int(Config.segcam_image_y)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

        pygame.init()
        pygame.font.init()

        self.client = carla.Client(Config.host, Config.port)
        self.client.set_timeout(60.0)

        self.control = None
        self.display = None
        self.scenario = None
        self.speed = None
        self.distance = None
        self.clock = pygame.time.Clock()

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
        wld.tick()

        self.episodes = list()
        for scenario in Config.scenarios:
            for speed in np.arange(Config.ped_speed_range[0], Config.ped_speed_range[1] + 1, 0.1):
                for distance in np.arange(Config.ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                    self.episodes.append((scenario, speed, distance))

    def reset(self):
        scenario_id, ped_speed, ped_distance = random.choice(self.episodes)
        self.scenario = scenario_id
        self.speed = ped_speed
        self.distance = ped_distance
        func = 'self.scene_generator.scenario' + scenario_id
        scenario = eval(func + '()')
        self.world.restart(scenario, ped_speed, ped_distance)
        self.planner_agent.update_scenario(scenario)
        self.client.get_world().tick()
        return self._get_observation()

    def _get_observation(self):
        control, observation = self.planner_agent.run_step()
        self.control = control
        return observation

    def step(self, action):
        self.world.tick(self.clock)
        if action == 0:
            self.control.throttle = 0.6
        elif action == 2:
            self.control.throttle = -0.6
        else:
            self.control.throttle = 0
        self.world.player.apply_control(self.control)
        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        observation = self._get_observation()
        reward, goal, accident, near_miss = self.planner_agent.get_reward()
        info = {"goal": goal, "accident": accident, "near miss": near_miss,
                "scenario": self.scenario, "ped_speed": self.speed, "ped_distance": self.distance}

        return observation, reward, goal or accident, info

    def render(self, mode="human"):
        if self.display is None:
            self.display = pygame.display.set_mode(
                (Config.width, Config.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0, 0, 0))
            pygame.display.flip()
        self.world.render(self.display)
        pygame.display.flip()

    def close(self):
        self.world.destroy()
        pygame.quit()


def main():
    env = GIDASBenchmark()

    for episodes in range(10):
        obs = env.reset()
        total_reward = 0
        for i in range(500):
            # env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print(info)
                print("Reward: {:.4f}".format(total_reward))
                break
    env.close()


if __name__ == '__main__':
    main()
