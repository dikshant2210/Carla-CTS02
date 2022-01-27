"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""

import carla
from agents.navigation.hylear_controller import HyLEAR


class ISDespotP(HyLEAR):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(ISDespotP, self).__init__(world, carla_map, scenario, conn, eval_mode=False)

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        # Steering action on the basis of shortest and safest path(Hybrid A*)
        obstacles = self.get_obstacles(start)
        if len(obstacles):
            path = self.get_path_simple(start, end, obstacles)
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
