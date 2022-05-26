"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""
import math
import operator
import random
import time
import carla
import numpy as np
from multiprocessing import Process, Pool
from collections import deque
import subprocess
from config import Config

from agents.navigation.rlagent import RLAgent
from ped_path_predictor.m2p3 import PathPredictor
from agents.navigation.risk_aware_path import PathPlanner


def run_server():
    subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car {}'.format(
        Config.despot_port)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


class HyLEAR(RLAgent):
    def __init__(self, world, carla_map, scenario, conn=None, eval_mode=False, agent='hylear'):
        super(HyLEAR, self).__init__(world, carla_map, scenario)

        self.conn = conn
        self.eval_mode = eval_mode
        self.agent = agent
        if not self.eval_mode:
            p = Process(target=run_server)
            p.start()
            self.conn.establish_connection()
            m = self.conn.receive_message()
            print(m)  # RESET
        self.ped_pred = PathPredictor("ped_path_predictor/_out/m2p3_289271.pth")
        self.ped_pred.model.eval()
        self.risk_path_planner = PathPlanner()

        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        sidewalk_cost = 50.0
        self.risk_cmp[4:7, 4:] = sidewalk_cost
        self.risk_cmp[:, 4:7] = sidewalk_cost
        self.risk_cmp[13:16, 13:] = sidewalk_cost
        self.risk_cmp[94:97, 13:] = sidewalk_cost
        self.risk_cmp[103:106, 13:] = sidewalk_cost
        self.risk_cmp[13:16, 16:94] = sidewalk_cost

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
        elif not self.pedestrian_observable:
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

        # t = time.time()
        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        if len(obstacles):
            if self.agent == 'hypal':
                path, intention = self.get_path_simple(start, end, obstacles)
            else:
                path, intention = self.get_path_with_reasoning(start, end, obstacles)
        else:
            path, intention = self.get_path_simple(start, end, obstacles)
        # print("time taken: ", time.time() - t)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.
        # print("Angle: ", control.steer)

        # Best speed action for the given path
        if not self.eval_mode:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, intention

    def get_path_simple(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]
        path = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                          yaw, self.risk_cmp, False)
        # path = self.find_path(start, end, self.grid_cost, obstacles)
        intention = self.get_car_intention([], path[0], start)
        return path, intention

    def get_path_with_reasoning(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]
        relaxed_sidewalk = self.grid_cost.copy()
        y = round(start[1])
        # Relax sidewalk
        sidewalk_cost = -1.0
        sidewalk_length = 20
        if self.scenario[0] in [1, 3, 4, 7, 8, 10]:
            relaxed_sidewalk[13:16, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[4:7, y - 10: sidewalk_length + sidewalk_length] = sidewalk_cost
        elif self.scenario[0] in [2, 5, 6, 9]:
            relaxed_sidewalk[94:97, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost
            relaxed_sidewalk[103:106, y - sidewalk_length: y + sidewalk_length] = sidewalk_cost

        if len(self.ped_history) < 15:
            path_normal = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                     yaw, self.risk_cmp, False)
            if path_normal[1] < 100:
                return path_normal[0], self.get_car_intention([], path_normal[0], start)
            paths = [path_normal,
                     self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                                yaw, self.risk_cmp, True)]  # Sidewalk relaxed
            path, _ = min(paths, key=lambda t: t[1])
            return path, self.get_car_intention([], path, start)
        else:
            # Use path predictor
            ped_updated_risk_cmp = self.risk_cmp.copy()
            ped_path = np.array(self.ped_history)
            ped_path = ped_path.reshape((15, 2))
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            new_obs = [obs for obs in obstacles]
            pedestrian_path_d = list()
            for node in pedestrian_path:
                if (round(node[0]), round(node[1])) not in new_obs:
                    new_obs.append((round(node[0]), round(node[1])))
                    pedestrian_path_d.append((round(node[0]), round(node[1])))
            for pos in new_obs:
                ped_updated_risk_cmp[pos[0] + 10, pos[1] + 10] = 10000

            path_normal = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                                     yaw, ped_updated_risk_cmp, False)
            if path_normal[1] < 100:
                # print("normal!", path_normal[1], (path_normal[0][2][2] - path_normal[0][1][2]) / 70.0)
                return path_normal[0], self.get_car_intention(pedestrian_path_d, path_normal[0], start)
            # print(start, end, obstacles)
            paths = [path_normal,  # Normal
                     self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, new_obs, car_speed,
                                                                yaw, ped_updated_risk_cmp, True),  # ped pred
                     self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, obstacles, car_speed,
                                                                yaw, ped_updated_risk_cmp, True),  # Sidewalk relaxed
                     self.risk_path_planner.find_path_with_risk(start, end, relaxed_sidewalk, new_obs, car_speed,
                                                                yaw, ped_updated_risk_cmp, True)]  # Sidewalk relaxed + ped pred
            path = self.rulebook(paths, start)
            # print(path[2][2] - start[2])
            return path, self.get_car_intention(pedestrian_path_d, path, start)

    @staticmethod
    def rulebook(paths, start):
        # No sidewalk
        data = []
        steer = []
        r = []
        for p in paths:
            path, risk = p
            len_path = len(path)
            if len_path == 0:
                lane = math.inf
            else:
                lane = sum([path[i][2] - path[i-1][2] for i in range(1, len_path)]) / len_path
            data.append((path, risk, lane, len_path))
            # r.append(risk)
            # steer.append((path[2][2] - start[2]) / 70.)

        # print("Rulebook!", r)
        # print("Steering angle: ", steer)
        data.sort(key=operator.itemgetter(1, 2, 3))
        path = data[0][0]
        # print("Steering angle: ", (path[2][2] - start[2]) / 70.)
        return data[0][0]
