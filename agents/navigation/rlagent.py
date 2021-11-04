"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import sys
import datetime
import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from agents.navigation.agent import Agent
from config import Config
from assets.occupancy_grid import OccupancyGrid
from agents.navigation.hybridastar import HybridAStar
# from agents.path_planner.hybridastar import HybridAStar


class RLAgent(Agent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, scenario):
        super(RLAgent, self).__init__(world.player)
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.scenario = scenario
        self.occupancy_grid = OccupancyGrid()
        self.fig = plt.figure()
        self.display_costmap = False
        self.prev_action = None
        self.folder = datetime.datetime.now().timestamp()
        # os.mkdir("_out/{}".format(self.folder))

        wps = carla_map.generate_waypoints(Config.grid_size)
        print("Total no. of waypoints: {}".format(len(wps)))

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
        self.vehicle_length = self.vehicle.bounding_box.extent.x * 2
        self.vehicle_width = self.vehicle.bounding_box.extent.y * 2
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y,
                                        obstacle, self.vehicle_length)

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

    def update_scenario(self, scenario):
        self.scenario = scenario

    def in_hit_area(self, x, y, theta, ped_x, ped_y):
        # TOP RIGHT VERTEX:
        top_right_x = x + ((self.vehicle_width / 2) * np.cos(theta)) - ((self.vehicle_length / 2) * np.sin(theta))
        top_right_y = y + ((self.vehicle_width / 2) * np.sin(theta)) + ((self.vehicle_length / 2) * np.cos(theta))

        # TOP LEFT VERTEX:
        top_left_x = x - ((self.vehicle_width / 2) * np.cos(theta)) - ((self.vehicle_length / 2) * np.sin(theta))
        top_left_y = y - ((self.vehicle_width / 2) * np.sin(theta)) + ((self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x - ((self.vehicle_width / 2) * np.cos(theta)) + ((self.vehicle_length / 2) * np.sin(theta))
        bot_left_y = y - ((self.vehicle_width / 2) * np.sin(theta)) - ((self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x + ((self.vehicle_width / 2) * np.cos(theta)) + ((self.vehicle_length / 2) * np.sin(theta))
        bot_right_y = y + ((self.vehicle_width / 2) * np.sin(theta)) - ((self.vehicle_length / 2) * np.cos(theta))

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]
        return 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

    def in_near_miss(self, x, y, theta, ped_x, ped_y):
        # TOP RIGHT VERTEX:
        top_right_x = x + ((0.5 + self.vehicle_width / 2) * np.cos(theta)) - \
                      ((1.5 + self.vehicle_length / 2) * np.sin(theta))
        top_right_y = y + ((0.5 + self.vehicle_width / 2) * np.sin(theta)) + \
                      ((1.5 + self.vehicle_length / 2) * np.cos(theta))

        # TOP LEFT VERTEX:
        top_left_x = x - ((0.5 + self.vehicle_width / 2) * np.cos(theta)) - \
                     ((1.5 + self.vehicle_length / 2) * np.sin(theta))
        top_left_y = y - ((0.5 + self.vehicle_width / 2) * np.sin(theta)) + \
                     ((1.5 + self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x - ((0.5 + self.vehicle_width / 2) * np.cos(theta)) + \
                     ((0.5 + self.vehicle_length / 2) * np.sin(theta))
        bot_left_y = y - ((0.5 + self.vehicle_width / 2) * np.sin(theta)) - \
                     ((0.5 + self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x + ((0.5 + self.vehicle_width / 2) * np.cos(theta)) + \
                      ((0.5 + self.vehicle_length / 2) * np.sin(theta))
        bot_right_y = y + ((0.5 + self.vehicle_width / 2) * np.sin(theta)) - \
                      ((0.5 + self.vehicle_length / 2) * np.cos(theta))

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]
        return 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

    def get_reward(self):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        goal = False
        near_miss = False
        hit = False
        if goal_dist < 3:
            goal = True
            return 1000, goal, hit, near_miss

        # Pedestrian hit and near miss section
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        # In hit area
        hit = self.in_hit_area(start[0], start[1], start[2], walker_x, walker_y)
        if hit:  # accident
            hit = True
            return -1000, goal, hit, near_miss
        # in near miss area
        near_miss = self.in_near_miss(start[0], start[1], start[2], walker_x, walker_y)

        # TODO: Collision with incoming or static car

        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            x = self.world.incoming_car.get_location().x
            y = self.world.incoming_car.get_location().y
            grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        if self.scenario[0] in [1, 2, 3, 6, 9, 10]:
            x = round(start[0] - self.min_x - 1)
        else:
            x = round(start[0] - self.min_x)
        location = [min(x, self.grid_cost.shape[0] - 1),
                    min(round(start[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        reward = -grid[location[0], location[1]]

        reward += -0.1
        if self.prev_action.throttle != 0:
            reward += -0.1
        if self.prev_action.steer != 0:
            reward += -1
        if near_miss:  # near miss
            reward += -500
        return reward, goal, hit, near_miss

    def get_car_intention(self, obstacles, path, start):
        return self.world.semseg_sensor.array.copy()
        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(self.world.semseg_sensor.array, file)
        costmap = self.grid_cost.copy()
        for node in path:
            i = round(node[0]) - self.min_x
            j = round(node[1]) - self.min_y
            costmap[j, i] = 0.0

        for obs in obstacles:
            i = round(obs[0])
            j = round(obs[1])
            costmap[i, j] = 10000

        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(costmap, file)

        idx1 = np.where(costmap == 10000)
        costmap[idx1] = 256.0
        costmap = 256.0 - costmap
        idx2 = np.where(costmap == 256)
        costmap[idx2] = 100.0

        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(costmap, file)

        x = round(start[0]) - self.min_x
        y = round(start[1]) - self.min_y
        boundaries = costmap.shape
        x1 = max(0, x - 50)
        x2 = min(boundaries[0], x + 50)
        y1 = max(0, y - 50)
        y2 = min(boundaries[1], y + 50)
        pad_y = 100 - (y2 - y1)
        pad_x = 100 - (x2 - x1)
        # print(pad_x, pad_y)
        return np.pad(costmap[y1:y2, x1:x2], ((round(pad_y / 2), pad_y - round(pad_y / 2)),
                                              (round(pad_x / 2), pad_x - round(pad_x / 2))), constant_values=0)

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 50.0:
            if self.scenario[0] == 3 and walker_x >= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
            elif self.scenario[0] in [7, 8] and walker_x <= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
            elif self.scenario[0] in [1, 4, 10]:
                obstacles.append((int(walker_x), int(walker_y)))
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                obstacles.append((int(car_x), int(car_y)))
        t0 = time.time()
        if self.scenario[0] in [1, 2, 3, 6, 9, 10]:
            car_lane = "right"
        else:
            car_lane = "left"
        paths = self.path_planner.find_path(start, end, self.grid_cost, obstacles, car_lane=car_lane)
        if len(paths):
            path = paths[0]
        else:
            path = []
        # print("Time taken to generate path {:.4f}ms".format((time.time() - t0) * 1000))
        path.reverse()

        if self.display_costmap:
            self.plot_costmap(obstacles, path)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.

        self.prev_action = control
        return control, self.get_car_intention(obstacles, path, start)

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
