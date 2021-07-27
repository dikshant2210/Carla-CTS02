"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from agents.navigation.mapinfo import MapInfo
from agents.navigation.car import Car
from agents.navigation.hybrid_a_star import HybridAStar
from agents.navigation.agent import Agent
from agents.navigation.config import Config


class HyLEAR(Agent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, conn):
        super(HyLEAR, self).__init__(world.player)
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.conn = conn
        self.folder = datetime.datetime.now().timestamp()
        os.mkdir("_out/{}".format(self.folder))

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

        self.max_x += 50
        self.max_y += 50
        self.min_y -= 50
        self.min_x -= 50
        self.costmap = np.zeros((int(self.max_x - self.min_x), int(self.max_y - self.min_y)))

        for wp in wps:
            exit_xyz = wp.transform.location
            x2, y2, z2 = np.round([exit_xyz.x, exit_xyz.y, exit_xyz.z], 0)
            x2 = int(x2 - self.min_x)
            y2 = int(y2 - self.min_y)
            self.costmap[x2, y2] = 1

            if wp.lane_type == 'Sidewalk':
                print(wp)

        self.costmap = gaussian_filter(self.costmap, sigma=4)
        self.costmap = self.costmap > 0.01
        self.costmap = self.costmap.astype(float)

    def get_costmap(self):
        # Update Costmap with latest info
        pedestrian_position = self.world.walker.get_location()
        updated_costmap = np.copy(self.costmap)
        x = int(pedestrian_position.x - self.min_x)
        y = int(pedestrian_position.y - self.min_y)
        updated_costmap[x, y] = 0
        x_limit = int(self.world.walker.bounding_box.extent.x)
        y_limit = int(self.world.walker.bounding_box.extent.y)

        for i in range(1, x_limit + 1):
            for j in range(1, y_limit + 1):
                updated_costmap[x + i, y + j] = 0
                updated_costmap[x - i, y + j] = 0
                updated_costmap[x + i, y - j] = 0
                updated_costmap[x - i, y - j] = 0
        return updated_costmap

    def get_path(self, cmp):
        minfo = MapInfo(cmp.shape[0], cmp.shape[1])
        car = Car(int(self.vehicle.bounding_box.extent.x * 2 + 1), int(self.vehicle.bounding_box.extent.y * 2 + 1))

        start = (self.vehicle.get_location().x - self.min_x, self.vehicle.get_location().y - self.min_y, 0)
        end = (2 - self.min_x, 150 - self.min_y, 0)

        epsilon = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        if epsilon < 10:
            print("Goal reached!")
            return [], [], True

        minfo.start = start
        minfo.end = end
        minfo.obstacle = np.argwhere(cmp == 0)
        car.set_position(minfo.start)
        plan = HybridAStar(minfo.start, minfo.end, minfo, car, r=2.0)
        flag = plan.run()
        steering = list()
        if flag:
            xs, ys, yaws = plan.reconstruct_path()
            steering = yaws
            minfo.path = list(zip(xs, ys))
        # waypoint_trajectory = list()
        # for loc in minfo.path:
        #     wp = self.wmap.get_waypoint(carla.Location(loc[0] + self.min_x, loc[1] + self.min_y, 0.1))
        #     if len(waypoint_trajectory) != 0 and waypoint_trajectory[-1] == wp:
        #         continue
        #     else:
        #         waypoint_trajectory.append(wp)

        return minfo.path, steering, False

    def run_step(self, debug=False):
        cmp = self.get_costmap()
        path, yaws, flag = self.get_path(cmp)
        if flag:
            return "goal"

        self.save_costmap(path, cmp)
        del cmp

        # print(self.vehicle.get_location())
        steering = np.clip(yaws[1] / Config.max_steering_angle, -1, 1)
        # print(steering, yaws)

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = 0.5
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def save_costmap(self, path, costmap):
        for wp in path:
            # loc = wp.transform.location
            # x, y = int(loc.x - self.min_x), int(loc.y - self.min_y)
            x, y = int(wp[0]), int(wp[1])
            costmap[x, y] = 0.5

        ts = datetime.datetime.now().timestamp()
        costmap = np.flip(np.rot90(costmap, axes=(1, 0)), axis=1)
        plt.imsave("_out/{}/{}.jpg".format(self.folder, ts), costmap, cmap='gray')
        plt.close()

    def update_information(self):
        pass
