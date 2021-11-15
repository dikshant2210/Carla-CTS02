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

    def get_reward(self, action):
        reward = 0
        goal = False
        hit = False
        nearmiss = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if speed > 0.3:
            scaling = self.linmap(0, Config.max_speed, 0, 1, min(speed, Config.max_speed))
            hit = self.in_hit_area(start[0], start[1], start[2], walker_x, walker_y)
            if hit:
                collision_reward = Config.hit_penalty * (scaling + 0.1)
                reward -= collision_reward

            nearmiss = self.in_near_miss(start[0], start[1], start[2], walker_x, walker_y)
            if nearmiss:
                nearmiss_reward = 500 * (scaling + 0.1)
                reward -= nearmiss_reward

        reward -= pow(goal_dist / 4935.0, 0.8) * 1.2

        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            x = self.world.incoming_car.get_location().x
            y = self.world.incoming_car.get_location().y
            grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # Penalizing for hitting an obstacle
        if self.scenario[0] in [1, 2, 3, 6, 9, 10]:
            x = round(start[0] - self.min_x - 1)
        else:
            x = round(start[0] - self.min_x)
        location = [min(x, self.grid_cost.shape[0] - 1),
                    min(round(start[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        obstacle_cost = grid[location[0], location[1]]
        if obstacle_cost <= 100:
            reward -= (obstacle_cost / 20.0)
        elif obstacle_cost <= 150:
            reward -= (obstacle_cost / 15.0)
        elif obstacle_cost <= 200:
            reward -= (obstacle_cost / 10.0)
        else:
            reward -= (obstacle_cost / 0.22)

        # "Heavily" penalize braking if you are already standing still
        if self.prev_speed is not None:
            if action == 2 and self.prev_speed < 0.2:
                reward -= 1

        # Penalize braking/acceleration actions to get a smoother ride
        if action != 0:
            reward -= 0.05

        reward -= pow(abs(self.prev_action.steer), 1.3) / 2.0

        if goal_dist < 3:
            reward += Config.goal_reward
            goal = True

        # Normalize reward
        reward = reward / 1000.0
        return reward, goal, hit, nearmiss

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
        ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y, front_margin=15, side_margin=0.5)
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
