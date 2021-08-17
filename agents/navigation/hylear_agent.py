"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import sys
import os
import datetime

import numpy as np

from agents.navigation.agent import Agent
from agents.navigation.config import Config
from assets.occupancy_grid import OccupancyGrid
from agents.navigation.hybridastar import HybridAStar


class HyLEAR(Agent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, conn):
        super(HyLEAR, self).__init__(world.player)
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.conn = conn
        self.occupancy_grid = OccupancyGrid()
        self.folder = datetime.datetime.now().timestamp()
        os.mkdir("_out/{}".format(self.folder))

        wps = carla_map.generate_waypoints(Config.grid_size)
        print("Total no. of waypoints: {}".format(len(wps)))
        self.conn.establish_connection()
        m = self.conn.receive_message()
        print(m)  # START

        self.max_x = -sys.maxsize - 1
        self.max_y = -sys.maxsize - 1
        self.min_x = sys.maxsize
        self.min_y = sys.maxsize

        for wp in wps:
            xyz = wp.transform.location
            self.max_x = max([self.max_x, xyz.x])
            self.max_y = max([self.max_y, xyz.y])
            self.min_x = min([self.min_x, xyz.x])
            self.min_y = min([self.min_y, xyz.y])

        # print(self.max_y, self.max_x, self.min_y, self.min_x)

        self.max_x = int(self.max_x)
        self.max_y = int(self.max_y)
        self.min_y = int(self.min_y)
        self.min_x = int(self.min_x)

        obstacle = []
        resolution = 1
        vehicle_length = 2
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y,
                                        obstacle, resolution, vehicle_length)

    def get_reward(self):
        return -1

    def run_step(self, debug=False):
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, -90)
        end = (2 - self.min_x, 150 - self.min_y, -90)

        dist_remaining = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)

        terminal = False
        # Handle reached goal situation
        # if dist_remaining < Config.location_threshold:
        #     terminal = True
        #     return True

        transform = self.vehicle.get_transform()
        reward = self.get_reward()
        angle = transform.rotation.yaw
        car_pos = [transform.location.x, transform.location.y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x**2 + car_velocity.y**2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]
        # print(terminal, reward, angle, car_pos, car_speed, pedestrian_positions)

        # print(self.min_x, self.max_x, self.min_y, self.max_y, start, end)
        path = self.path_planner.find_path(start, end, self.occupancy_grid)
        self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
        m = self.conn.receive_message()
        acc = 0
        if m[0] == '0':
            acc = 5
        elif m[0] == '2':
            acc = -5
        # print("Speed action: {}".format(acc))

        control = carla.VehicleControl()
        control.steer = path[1][2] - path[0][2]
        control.throttle = acc
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control
