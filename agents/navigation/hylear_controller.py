"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""
import multiprocessing
import time

import carla
import numpy as np
from multiprocessing import Process, Pool
from collections import deque
import subprocess

from agents.navigation.rlagent import RLAgent
from ped_path_predictor.m2p3 import PathPredictor
from agents.navigation.risk_aware_path import PathPlanner


def run_server():
    subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


class HyLEAR(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None, eval_mode=False):
        super(HyLEAR, self).__init__(world, carla_map, scenario)

        self.conn = conn
        self.eval_mode = eval_mode
        if not self.eval_mode:
            p = Process(target=run_server)
            p.start()
            self.conn.establish_connection()
            m = self.conn.receive_message()
            print(m)  # RESET
        self.ped_history = deque(list(), maxlen=15)
        self.ped_pred = PathPredictor("ped_path_predictor/_out/m2p3_70797.pth")
        self.ped_pred.model.eval()
        self.risk_path_planner = PathPlanner()
        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        self.risk_cmp[4:7, 4:] = 50.0
        self.risk_cmp[:, 4:7] = 50.0
        self.risk_cmp[13:16, 13:] = 50.0
        self.risk_cmp[94:97, 13:] = 50.0
        self.risk_cmp[103:106, 13:] = 50.0
        self.risk_cmp[13:16, 16:94] = 50.0

    def update_scenario(self, scenario):
        super(HyLEAR, self).update_scenario(scenario)
        self.ped_history = deque(list(), maxlen=15)

    def get_reward_despot(self, action):
        base_reward, goal, hit, nearmiss, terminal = super(HyLEAR, self).get_reward(action)
        reward = 0
        if goal:
            reward += 1.0
        return reward, goal, hit, nearmiss, terminal

    def get_speed_action(self, path, control):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y

        if self.prev_action is not None:
            reward, goal, hit, near_miss, terminal = self.get_reward_despot(self.prev_speed)
            terminal = goal or hit
        else:
            # handling first instance
            reward = 0
            terminal = False
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]

        if len(path) == 0:
            control.brake = 0.6
            self.prev_speed = 2
        elif np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) > 50.0:
            control.throttle = 0.6
        else:
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            if m == "START":
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
        return control

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        if len(obstacles):
            path = self.get_path_with_reasoning(start, end, obstacles)
        else:
            path = self.get_path_simple(start, end, obstacles)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.

        # Best speed action for the given path
        if not self.eval_mode:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, self.get_car_intention(obstacles, path, start)

    def get_path_simple(self, start, end, obstacles):
        path = self.find_path(start, end, self.grid_cost, obstacles)
        return path

    def get_path_ped_prediction(self, start, end, obstacles):
        if len(self.ped_history) == 15:
            # Use path predictor
            ped_path = np.array(self.ped_history)
            ped_path = ped_path.reshape((1, 15, 2))
            pedestrian_path = self.ped_pred.get_pred(ped_path)
            for node in pedestrian_path[0]:
                if (round(node[0]), round(node[1])) not in obstacles and round(node[0]) <= obstacles[0][0]:
                    obstacles.append((round(node[0]), round(node[1])))
        print(obstacles, start)
        costmap = self.grid_cost.copy()
        y = round(start[1])
        costmap[13:16, y - 20: y + 20] = 0
        costmap[4:7, y - 20: y + 20] = 0
        path = self.find_path(start, end, costmap, obstacles)
        return path

    def get_path_with_reasoning(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        yaw = self.vehicle.get_transform().rotation.yaw
        relaxed_sidewalk = self.grid_cost.copy()
        y = round(start[1])
        # Relax sidewalk
        relaxed_sidewalk[13:16, y - 20: y + 20] = 0
        relaxed_sidewalk[4:7, y - 20: y + 20] = 0

        if len(self.ped_history) < 15 or True:
            try:
                paths = [self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                    yaw, self.risk_cmp),  # Normal
                         self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                                    yaw, self.risk_cmp)]  # Sidewalk relaxed
                path, _ = min(paths, key=lambda t: t[1])
                return path
            except:
                return []

        else:
            # Use path predictor
            ped_updated_risk_cmp = self.risk_cmp.copy()
            ped_path = np.array(self.ped_history)
            ped_path = ped_path.reshape((15, 2))
            t0 = time.time()
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            time_taken = (time.time() - t0) * 1000
            new_obs = [obs for obs in obstacles]
            for node in pedestrian_path:
                if (round(node[0]), round(node[1])) not in new_obs:
                    new_obs.append((round(node[0]), round(node[1])))
            for pos in new_obs:
                ped_updated_risk_cmp[pos[0], pos[1]] = 1000
            params = [[start, end, self.grid_cost, obstacles, car_speed, yaw, self.risk_cmp],
                      [start, end, relaxed_sidewalk, obstacles, car_speed, yaw, self.risk_cmp],
                      [start, end, self.grid_cost, new_obs, car_speed, yaw, self.risk_cmp],
                      [start, end, relaxed_sidewalk, new_obs, car_speed, yaw, self.risk_cmp]]

            t0 = time.time()
            try:
                with multiprocessing.Pool(processes=len(params)) as pool:
                    paths = pool.starmap(self.risk_path_planner.find_path_with_risk, params)
                    path = min(paths, key=lambda t: t[1])
                    p = path[0]
            except:
                p = []
            print("Time taken(with prediction): {:.4f}ms, Prediction time: {:.4f}ms".format((time.time() - t0) * 1000,
                                                                                            time_taken))
            return p

    def get_obstacles(self, start):
        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 50.0:
            self.ped_history.append([walker_x, walker_y])
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

        return obstacles
