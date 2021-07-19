"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.mapinfo import MapInfo
from agents.navigation.car import Car
from agents.navigation.hybrid_a_star import HybridAStar


class HyLEAR(object):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map):
        self.world = world
        self.vehicle = world.player
        self._sampling_resolution = 4.5

        self.dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution=self._sampling_resolution)
        self.topology, self.max_x, self.max_y, self.min_x, self.min_y = self.dao.get_topology()
        self.max_x += 50
        self.max_y += 50
        self.min_y -= 50
        self.min_x -= 50
        self.costmap = np.zeros((int(self.max_x - self.min_x), int(self.max_y - self.min_y)))

        for wp in carla_map.generate_waypoints(2):
            exit_xyz = wp.transform.location
            x2, y2, z2 = np.round([exit_xyz.x, exit_xyz.y, exit_xyz.z], 0)
            x2 = int(x2 - self.min_x)
            y2 = int(y2 - self.min_y)
            self.costmap[x2, y2] = 1
            if wp.lane_type == 'Sidewalk':
                print(wp)

        self.costmap = gaussian_filter(self.costmap, sigma=4)
        self.costmap = self.costmap > 0.01
        self.path = self.get_path()

    def get_costmap(self):
        # Update Costmap with latest info
        return self.costmap

    def get_path(self):
        cmp = self.get_costmap()
        minfo = MapInfo(cmp.shape[0], cmp.shape[1])
        car = Car(int(self.vehicle.bounding_box.extent.x * 2 + 1), int(self.vehicle.bounding_box.extent.y * 2 + 1))
        print(int(self.vehicle.bounding_box.extent.x * 2 + 1), int(self.vehicle.bounding_box.extent.y * 2 + 1))

        start = (2 - self.min_x, 234 - self.min_y, 0)
        end = (2 - self.min_x, 100 - self.min_y, 0)

        minfo.start = start
        minfo.end = end
        minfo.obstacle = np.argwhere(cmp == 0)
        car.set_position(minfo.start)
        plan = HybridAStar(minfo.start, minfo.end, minfo, car, r=5.0)
        if plan.run():
            xs, ys, yaws = plan.reconstruct_path()
            minfo.path = list(zip(xs, ys))

        return minfo.path

    def run_step(self):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.5
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def update_information(self):
        pass
