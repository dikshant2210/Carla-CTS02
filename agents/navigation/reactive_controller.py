"""
Author: Dikshant Gupta
Time: 15.11.21 16:32
"""

import carla
import numpy as np

from config import Config
from agents.navigation.rlagent import RLAgent


class ReactiveController(RLAgent):
    def __init__(self, world, carla_map, scenario):
        super(ReactiveController, self).__init__(world, carla_map, scenario)

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

        # Best speed action on the basis of reactive controller
        velocity = self.vehicle.get_velocity()
        self.prev_speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6
        ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                    front_margin=10, side_margin=0.5)
        # if ped_hit:
        #     _ = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                           front_margin=0, side_margin=0, debug=True)
        #     print("Walker ", walker_x, walker_y)
        #     print(" Car ", start[0], start[1], start[2])
        if self.prev_speed > Config.max_speed or ped_hit:
            control.brake = 0.6
        else:
            control.throttle = 0.6

        self.prev_action = control
        return control, self.get_car_intention(obstacles, path, start)
