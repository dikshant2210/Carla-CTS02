"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""

import carla
import time
import numpy as np
from multiprocessing import Process
import subprocess
from config import Config

from agents.navigation.rlagent import RLAgent


def run_server():
    subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


class ISDespotP(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(ISDespotP, self).__init__(world, carla_map, scenario)

        self.conn = conn
        p = Process(target=run_server)
        p.start()
        self.conn.establish_connection()
        m = self.conn.receive_message()
        print(m)  # START

    def get_reward(self, action):
        reward = -0.1
        goal = False

        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if goal_dist < 3:
            reward += Config.goal_reward
            goal = True

        return reward, goal, False, False

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # Steering action on the basis of shortest and safest path(Hybrid A*)
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
        path.reverse()

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.

        # Best speed action for the given path
        if self.prev_action is not None:
            reward, goal, hit, near_miss = self.get_reward(self.prev_speed)
            terminal = goal or hit
        else:
            # handling first instance
            reward = 0
            terminal = False
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]

        if len(path) == 0:
            control.throttle = -0.6
        else:
            t0 = time.time()
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            self.prev_speed = 1
            if m[0] == '0':
                control.throttle = 0.6
                self.prev_speed = 0
            elif m[0] == '2':
                control.brake = 0.6
                self.prev_speed = 2

        self.prev_action = control
        return control, self.get_car_intention(obstacles, path, start)
