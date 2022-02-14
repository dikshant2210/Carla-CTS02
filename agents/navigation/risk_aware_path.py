"""
Author: Dikshant Gupta
Time: 16.01.22 23:11
"""

import numpy as np

from agents.navigation.hybridastar import HybridAStar
from agents.tools.risk_assesment import PerceivedRisk


class PathPlanner:
    def __init__(self):
        self.min_x = -10
        self.max_x = 100
        self.min_y = -10
        self.max_y = 300
        self.vehicle_length = 4.18
        self.risk_estimator = PerceivedRisk()
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y, [], self.vehicle_length)

    def find_path(self, start, end, costmap, obstacles, car_speed):
        paths = self.path_planner.find_path(start, end, costmap, obstacles)
        if len(paths):
            path = paths[0]
        else:
            path = []
        path.reverse()
        return path

    def find_path_with_risk(self, start, end, costmap, obstacles, car_speed, yaw, risk_map):
        try:
            path = self.find_path(start, end, costmap, obstacles, car_speed / 3.6)
            if len(path):
                player = [start[0], start[1], car_speed, yaw]
                steering_angle = path[2][2] - start[2]
                risk, drf = self.risk_estimator.get_risk(player, steering_angle, risk_map)
            else:
                risk = np.inf
                # TODO: DRF in this case
        except:
            path, risk = [], np.inf
        return path, risk
