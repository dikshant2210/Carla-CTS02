"""
Author: Dikshant Gupta
Time: 11.11.21 15:48
"""

import gym
import numpy as np
import carla
import pygame
import random

from environment.world import World
from environment.hud import HUD
from agents.navigation.rlagent import RLAgent
from agents.navigation.reactive_controller import ReactiveController
from agents.navigation.isdespot import ISDespotP
from config import Config
from agents.tools.scenario import Scenario
from agents.tools.connector import Connector


class GIDASBenchmark(gym.Env):
    def __init__(self):
        super(GIDASBenchmark, self).__init__()
        random.seed(100)
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
        self.mode = "TRAINING"
        self._max_episode_steps = 500
        self.clock = pygame.time.Clock()

        hud = HUD(Config.width, Config.height)
        self.client.load_world('Town01_Opt')
        wld = self.client.get_world()
        wld.unload_map_layer(carla.MapLayer.StreetLights)
        wld.unload_map_layer(carla.MapLayer.Props)
        # wld.unload_map_layer(carla.MapLayer.Particles)
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

        self.test_episodes = None
        self.episodes = list()
        print(Config.scenarios)
        for scenario in Config.scenarios:
            for speed in np.arange(Config.ped_speed_range[0], Config.ped_speed_range[1] + 1, 0.1):
                for distance in np.arange(Config.ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                    self.episodes.append((scenario, speed, distance))

    def reset(self):
        scenario_id, ped_speed, ped_distance = self.next_scene()
        # ped_speed = 0  # Debug Settings
        # ped_distance = 12
        # scenario_id = "01"
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
            self.control.throttle = 0.7
        elif action == 2:
            self.control.brake = 0.6
        elif action == 1:
            self.control.throttle = 0
        self.world.player.apply_control(self.control)
        if Config.synchronous:
            frame_num = self.client.get_world().tick()

        observation = self._get_observation()
        reward, goal, accident, near_miss, terminal = self.planner_agent.get_reward(action)
        info = {"goal": goal, "accident": accident, "near miss": near_miss,
                "velocity": self.planner_agent.vehicle.get_velocity(),
                "scenario": self.scenario, "ped_speed": self.speed, "ped_distance": self.distance}

        if self.mode == "TESTING":
            terminal = goal
        return observation, reward, terminal, info

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

    def reset_agent(self, agent):
        if agent == 'reactive':
            self.planner_agent = ReactiveController(self.world, self.map, self.scene)
        if agent == 'isdespot':
            conn = Connector(Config.despot_port)
            self.planner_agent = ISDespotP(self.world, self.map, self.scene, conn)

    def eval(self, current_episode=0):
        self.mode = "TESTING"
        episodes = list()
        for scenario in Config.test_scenarios:
            for speed in np.arange(Config.test_ped_speed_range[0], Config.test_ped_speed_range[1] + 1, 0.1):
                for distance in np.arange(Config.test_ped_distance_range[0], Config.test_ped_distance_range[1] + 1, 1):
                    episodes.append((scenario, speed, distance))
        self.episodes = episodes[current_episode:]
        self.test_episodes = iter(episodes[current_episode:])

    def next_scene(self):
        if self.mode == "TRAINING":
            return random.choice(self.episodes)
        elif self.mode == "TESTING":
            scene_config = next(self.test_episodes)
            return scene_config
