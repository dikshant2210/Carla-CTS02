"""
Author: Dikshant Gupta
Time: 28.08.22 22:00
"""


import carla
import numpy as np

from agents.navigation.hylear_controller import HyLEAR


class A2CCadrl(HyLEAR):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(A2CCadrl, self).__init__(world, carla_map, scenario, conn, eval_mode=False, agent='cadrl')

    def get_reward(self, action):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        other_agents = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        other_agents.append((walker_x, walker_y))
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            other_agents.append((car_x, car_y))

        reward = 0
        if goal_dist < 3:
            reward = 1.0
        dmin = min([np.sqrt((start[0] - x[0]) ** 2 + (start[1] - x[1]) ** 2) for x in other_agents])
        if dmin < 0.2:
            reward = -0.1 + (dmin / 2)

        _, goal, hit, nearmiss, terminal = super(HyLEAR, self).get_reward(action)
        return reward, goal, hit, nearmiss, terminal

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        (path, risk), intention = self.get_path_simple(start, end, obstacles)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.

        self.prev_action = control
        return control, intention, risk, self.pedestrian_observable
