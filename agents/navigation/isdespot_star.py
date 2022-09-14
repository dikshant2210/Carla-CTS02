"""
Author: Dikshant Gupta
Time: 05.04.22 15:43
"""

import carla
import numpy as np
from agents.navigation.hylear_controller import HyLEAR


class ISDespotPStar(HyLEAR):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(ISDespotPStar, self).__init__(world, carla_map, scenario, conn, eval_mode=False)

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

        # Best speed action for the given path
        if not self.eval_mode:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, intention, risk, self.pedestrian_observable

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
            if len(self.ped_history) >= 15:
                ped_path = np.array(self.ped_history)
                ped_path = ped_path.reshape((15, 2))
                pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            else:
                pedestrian_path = None
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed,
                                   pedestrian_positions, path, pedestrian_path)
            m = self.conn.receive_message()
            if m == "START":
                self.conn.send_message(terminal, reward, angle, car_pos, car_speed,
                                       pedestrian_positions, path, pedestrian_path)
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
