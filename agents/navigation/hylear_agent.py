"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import sys
import os
import datetime
import math

import numpy as np
import matplotlib.pyplot as plt

from agents.navigation.agent import Agent
from agents.navigation.config import Config
from assets.occupancy_grid import OccupancyGrid
# from agents.navigation.hybridastar import HybridAStar
from agents.path_planner.hybridastar import HybridAStar


class HyLEAR(Agent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, conn, scenario):
        super(HyLEAR, self).__init__(world.player)
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.conn = conn
        self.scenario = scenario
        self.occupancy_grid = OccupancyGrid()
        self.fig = plt.figure()
        self.folder = datetime.datetime.now().timestamp()
        os.mkdir("_out/{}".format(self.folder))

        wps = carla_map.generate_waypoints(Config.grid_size)
        print("Total no. of waypoints: {}".format(len(wps)))
        # self.conn.establish_connection()
        # m = self.conn.receive_message()
        # print(m)  # START

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
        vehicle_length = 3
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y,
                                        obstacle, resolution, vehicle_length)

    def get_reward(self):
        return -1

    def run_step(self, debug=False):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        if self.vehicle.get_location().y < end[1]:
            print("Goal reached!")
            return "goal"

        terminal = False
        reward = self.get_reward()
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x**2 + car_velocity.y**2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]
        # print(terminal, reward, angle, car_pos, car_speed, pedestrian_positions)

        # print(self.min_x, self.max_x, self.min_y, self.max_y, start, end)
        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 10.0:
            obstacles.append((int(walker_x), int(walker_y)))

        # if self.scenario[0] > 9:
        #     walker_extent = self.world.walker.bounding_box.extent
        #     walker_wp = self.wmap.get_waypoint(self.world.walker.get_location())
        #     for x in range(-math.ceil(walker_extent.x), math.ceil(walker_extent.x)):
        #         for y in range(-math.ceil(walker_extent.y), math.ceil(walker_extent.y)):
        #             obstacles.append((walker_wp.transform.location.x + x, walker_wp.transform.location.y + y))
        #
        #     car_extent = self.world.incoming_car.bounding_box.extent
        #     car_wp = self.wmap.get_waypoint(self.world.incoming_car.get_location())
        #     for x in range(-math.ceil(car_extent.x), math.ceil(car_extent.x)):
        #         for y in range(-math.ceil(car_extent.y), math.ceil(car_extent.y)):
        #             obstacles.append((car_wp.transform.location.x + x, car_wp.transform.location.y + y))
        #
        # else:
        #     walker_extent = self.world.walker.bounding_box.extent
        #     walker_wp = self.wmap.get_waypoint(self.world.walker.get_location())
        #     for x in range(-math.ceil(walker_extent.x), math.ceil(walker_extent.x)):
        #         for y in range(-math.ceil(walker_extent.y), math.ceil(walker_extent.y)):
        #             obstacles.append((walker_wp.transform.location.x + x, walker_wp.transform.location.y + y))

        path = self.path_planner.find_path(start, end, self.occupancy_grid, obstacles)
        path.reverse()

        cp = self.occupancy_grid.get_costmap([])
        x, y = list(), list()
        for node in path:
            pixel_coord = self.occupancy_grid.map.convert_to_pixel(node)
            x.append(pixel_coord[0] + 340)
            y.append(pixel_coord[1])
        # print(cp.max(), cp.min())
        plt.plot(x, y, "-r")
        if len(obstacles) > 0:
            obstacle_pixel = self.occupancy_grid.map.convert_to_pixel([obstacles[0][0], obstacles[0][1], 0])
            plt.scatter([obstacle_pixel[0] + 340], [obstacle_pixel[1]], c="k")
        # print(obstacle_pixel)
        plt.imshow(cp, cmap="gray")
        plt.axis([300, 600, 950, 1550])
        plt.draw()
        plt.pause(0.1)
        self.fig.clear()

        # rx1, ry1, ox1, oy1 = list(), list(), list(), list()
        # for node in path:
        #     rx1.append(node[0])
        #     ry1.append(node[1])
        # plt.plot(rx1, ry1, "-b")
        # for loc in obstacles:
        #     ox1.append(loc[0])
        #     oy1.append(loc[1])
        # plt.plot(ox1, oy1, ".k")
        # plt.grid(True)
        # plt.axis("equal")
        # plt.savefig("_out/path_{}.png".format(len(path)))
        # plt.close()

        # self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
        # m = self.conn.receive_message()
        # acc = 0
        # if m[0] == '0':
        #     acc = 5
        # elif m[0] == '2':
        #     acc = -5
        # print("Speed action: {}".format(acc))

        control = carla.VehicleControl()
        control.steer = (path[2][2] - start[2]) / 70.
        control.throttle = 0.15
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        print(control.steer, path[2][2], start[2])

        # control = carla.VehicleControl()
        # control.steer = 0
        # control.throttle = 0.2
        # control.brake = 0.0
        # control.hand_brake = False
        # control.manual_gear_shift = False

        return control
