"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import sys
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from agents.navigation.agent import Agent
from agents.navigation.config import Config
from assets.occupancy_grid import OccupancyGrid
from agents.navigation.hybridastar import HybridAStar
# from agents.path_planner.hybridastar import HybridAStar


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
        self.display_costmap = False
        self.prev_action = None
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
        self.min_y = int(self.min_y) - 10
        self.min_x = int(self.min_x) - 10

        obstacle = []
        vehicle_length = self.vehicle.bounding_box.extent.x * 2
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y,
                                        obstacle, vehicle_length)

        x_range = self.max_x - self.min_x
        y_range = self.max_y - self.min_y
        self.grid_cost = np.ones((x_range, y_range))
        for i in range(self.min_x, self.max_x):
            for j in range(self.min_y, self.max_y):
                loc = self.occupancy_grid.map.convert_to_pixel([i, j, 0])
                x = loc[0]
                y = loc[1]
                if x > 2199:
                    x = 2199
                if y > 2599:
                    y = 2599
                val = self.occupancy_grid.static_map[x, y]
                if val == 50:
                    val = 50.0
                self.grid_cost[i - self.min_x, j - self.min_y] = val

    def get_reward(self):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        if goal_dist < 3:
            print("Goal reached!")
            return 1000, True
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        ped_dist = np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2)
        if ped_dist < 0.5:  # accident
            return -1000, True

        reward = -0.1
        # TODO: add sidewalk reward
        if self.prev_action.throttle != 0:
            reward += -0.1
        if self.prev_action.steer != 0:
            reward += -1
        elif 0.5 < ped_dist < 1.5:  # near miss
            reward += -500
        return -0.1, False

    def run_step(self, debug=False):
        # control = carla.VehicleControl()
        # control.steer = 0
        # control.throttle = 1.0
        # control.brake = 0.0
        # control.hand_brake = False
        # control.manual_gear_shift = False
        # return control

        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        if self.prev_action is not None:
            reward, terminal = self.get_reward()
        else:
            # handling first instance
            reward = 0
            terminal = False
        if terminal:
            return "goal"
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]
        # print(terminal, reward, angle, car_pos, car_speed, pedestrian_positions)

        # print(self.min_x, self.max_x, self.min_y, self.max_y, start, end)
        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 50.0:
            obstacles.append((int(walker_x), int(walker_y)))
            pedestrian_positions = [[walker_x, walker_y]]
        if self.scenario == 10:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                obstacles.append((int(car_x), int(car_y)))
        # print(obstacles)
        t0 = time.time()
        paths = self.path_planner.find_path(start, end, self.grid_cost, obstacles)
        if len(paths):
            path = paths[0]
        else:
            path = []
        print("Time taken to generate path {:.4f}ms".format((time.time() - t0) * 1000))
        path.reverse()

        if self.display_costmap:
            self.plot_costmap(obstacles, path)

        control = carla.VehicleControl()
        if len(path) == 0:
            control.throttle = -1
            control.steer = 0
        else:
            t0 = time.time()
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            acc = 0
            if m[0] == '0':
                acc = 0.6
            elif m[0] == '2':
                acc = -0.6
            # print("Time taken to generate speed action {:.4f}ms".format((time.time() - t0) * 1000))
            # print("Speed action: {}".format(acc))
            control.throttle = acc
            control.steer = (path[2][2] - start[2]) / 70.
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        # print(control.steer, path[2][2], start[2])
        self.prev_action = control
        return control

    def plot_costmap(self, obstacles, path):
        cp = self.occupancy_grid.get_costmap([])
        x, y = list(), list()
        for node in path:
            pixel_coord = self.occupancy_grid.map.convert_to_pixel(node)
            x.append(pixel_coord[0])
            y.append(pixel_coord[1])
        plt.plot(x, y, "-r")
        if len(obstacles) > 0:
            obstacle_pixel = self.occupancy_grid.map.convert_to_pixel([obstacles[0][0], obstacles[0][1], 0])
            plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        # print(obstacle_pixel)
        plt.imshow(cp, cmap="gray")
        plt.axis([0, 350, 950, 1550])
        plt.draw()
        plt.pause(0.1)
        self.fig.clear()
