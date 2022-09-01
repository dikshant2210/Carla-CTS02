"""
Author: Dikshant Gupta
Time: 28.08.22 22:00
"""


import carla
import numpy as np

from config import Config
from agents.navigation.hylear_controller import HyLEAR


class A2CCadrl(HyLEAR):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(A2CCadrl, self).__init__(world, carla_map, scenario, conn, eval_mode=False, agent='cadrl')

    def get_reward(self, action):
        # transform = self.vehicle.get_transform()
        # start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        # end = self.scenario[2]
        # goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        #
        # other_agents = list()
        # walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        # other_agents.append((walker_x, walker_y))
        # if self.scenario[0] in [3, 7, 8, 10]:
        #     car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
        #     other_agents.append((car_x, car_y))
        #
        # reward = -goal_dist / 1000
        # _, goal, hit, nearmiss, terminal = super(HyLEAR, self).get_reward(action)
        # dmin = min([np.sqrt((start[0] - x[0]) ** 2 + (start[1] - x[1]) ** 2) for x in other_agents])
        #
        # if hit:
        #     reward = -1.0
        # elif goal_dist < 3:
        #     reward += 1.0
        # elif dmin < 0.2:
        #     reward += -0.1 - (dmin / 2)
        #
        # return reward, goal, hit, nearmiss, terminal

        reward = 0
        goal = False
        terminal = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if speed > 1.0:
            other_agents = list()
            walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
            other_agents.append((walker_x, walker_y))
            if self.scenario[0] in [3, 7, 8, 10]:
                car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
                other_agents.append((car_x, car_y))

            reward = -goal_dist / 1000
            _, goal, hit, nearmiss, terminal = super(HyLEAR, self).get_reward(action)
            dmin = min([np.sqrt((start[0] - x[0]) ** 2 + (start[1] - x[1]) ** 2) for x in other_agents])
            collision_reward = -0.1 - (dmin / 2.0)
            reward -= collision_reward
            # if speed <= 20:
            #     ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
            #                                 front_margin=1, side_margin=0.5)
            # else:
            #     ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
            #                                 front_margin=2, side_margin=0.5)
            # if ped_hit:
            #     # scale penalty by impact speed
            #     # hit = True
            #     scaling = self.linmap(0, Config.max_speed, 0, 1, min(speed * 0.27778, Config.max_speed))  # in m/s
            #     collision_reward = Config.hit_penalty * (scaling + 0.1)
            #     # if collision_reward >= 700:
            #     #     terminal = True
            #     reward -= collision_reward

        reward -= pow(goal_dist / 4935.0, 0.8) * 1.2

        # All grid positions of incoming_car in player rectangle
        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            xmin = round(car_x - self.vehicle_width / 2)
            xmax = round(car_x + self.vehicle_width / 2)
            ymin = round(car_y - self.vehicle_length / 2)
            ymax = round(car_y + self.vehicle_length / 2)
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    grid[round(x), round(y)] = 100
            # print(xmin, xmax, ymin, ymax)
            # x = self.world.incoming_car.get_location().x
            # y = self.world.incoming_car.get_location().y
            # grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # Penalizing for hitting an obstacle
        location = [min(round(start[0] - self.min_x), self.grid_cost.shape[0] - 1),
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
            if action != 0 and self.prev_speed < 0.28:
                reward -= Config.braking_penalty

        # Limit max speed to 50
        if self.prev_speed is not None:
            if action == 0 and self.prev_speed > Config.max_speed:
                reward -= Config.braking_penalty

        # Penalize braking/acceleration actions to get a smoother ride
        if self.prev_action.brake > 0:
            last_action = 2
        elif self.prev_action.throttle > 0:
            last_action = 0
        else:
            last_action = 1
        if last_action != 1 and last_action != action:
            reward -= 0.05

        reward -= pow(abs(self.prev_action.steer), 1.3) / 2.0

        if goal_dist < 3:
            reward += Config.goal_reward
            goal = True
            terminal = True

        # Normalize reward
        reward = reward / 1000.0

        # hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                         front_margin=0.2, side_margin=0.2, back_margin=0.1) or obstacle_cost > 50.0
        # hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                         front_margin=0.01, side_margin=0.01, back_margin=0.01) or obstacle_cost > 50.0
        hit = self.world.collision_sensor.flag or obstacle_cost > 50.0
        nearmiss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=1.5, side_margin=0.5, back_margin=0.5)
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
