"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from agents.navigation.agent import Agent
from config import Config
from assets.occupancy_grid import OccupancyGrid
from agents.navigation.hybridastar import HybridAStar
# from agents.path_planner.hybridastar import HybridAStar


class RLAgent(Agent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, scenario):
        super(RLAgent, self).__init__(world.player)
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.scenario = scenario
        self.occupancy_grid = OccupancyGrid()
        self.fig = plt.figure()
        self.display_costmap = False
        self.prev_action = None
        self.prev_speed = None
        self.folder = datetime.datetime.now().timestamp()
        self.past_trajectory = []
        # os.mkdir("_out/{}".format(self.folder))

        obstacle = []
        self.vehicle_length = self.vehicle.bounding_box.extent.x * 2
        self.vehicle_width = self.vehicle.bounding_box.extent.y * 2

        self.grid_cost = np.ones((110, 310)) * 1000.0
        # Road Network
        self.grid_cost[7:13, 13:] = 1.0
        self.grid_cost[97:103, 13:] = 1.0
        self.grid_cost[7:, 7:13] = 1.0

        self.min_x = -10
        self.max_x = 100
        self.min_y = -10
        self.max_y = 300
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y, obstacle, self.vehicle_length)

    def update_scenario(self, scenario):
        self.scenario = scenario
        self.past_trajectory = []

    def in_rectangle(self, x, y, theta, ped_x, ped_y, front_margin=1.5, side_margin=0.5, back_margin=0.5, debug=False):
        theta = theta / (2 * np.pi)
        # TOP RIGHT VERTEX:
        top_right_x = x + ((side_margin + self.vehicle_width / 2) * np.sin(theta)) + \
                      ((front_margin + self.vehicle_length / 2) * np.cos(theta))
        top_right_y = y - ((side_margin + self.vehicle_width / 2) * np.cos(theta)) + \
                      ((front_margin + self.vehicle_length / 2) * np.sin(theta))

        # TOP LEFT VERTEX:
        top_left_x = x - ((side_margin + self.vehicle_width / 2) * np.sin(theta)) + \
                     ((front_margin + self.vehicle_length / 2) * np.cos(theta))
        top_left_y = y + ((side_margin + self.vehicle_width / 2) * np.cos(theta)) + \
                     ((front_margin + self.vehicle_length / 2) * np.sin(theta))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x - ((side_margin + self.vehicle_width / 2) * np.sin(theta)) - \
                     ((back_margin + self.vehicle_length / 2) * np.cos(theta))
        bot_left_y = y + ((side_margin + self.vehicle_width / 2) * np.cos(theta)) - \
                     ((back_margin + self.vehicle_length / 2) * np.sin(theta))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x + ((side_margin + self.vehicle_width / 2) * np.sin(theta)) - \
                      ((back_margin + self.vehicle_length / 2) * np.cos(theta))
        bot_right_y = y - ((side_margin + self.vehicle_width / 2) * np.cos(theta)) - \
                      ((back_margin + self.vehicle_length / 2) * np.sin(theta))

        if debug:
            print("Top Left ", top_left_x, top_left_y)
            print("Top Right ", top_right_x, top_right_y)
            print("Bot Left ", bot_left_x, bot_left_y)
            print("Bot Right ", bot_right_x, bot_right_y)

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]
        return 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

    def linmap(self, a, b, c, d, x):
        return (x - a) / (b - a) * (d - c) + c

    def get_reward(self, action):
        reward = 0
        goal = False
        hit = False
        terminal = False
        nearmiss = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if speed > 0.3:
            if speed <= 20:
                ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=1, side_margin=0.75)
            else:
                ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=2, side_margin=1.2)
            if ped_hit:
                # scale penalty by impact speed
                hit = True
                scaling = self.linmap(0, Config.max_speed, 0, 1, min(speed, Config.max_speed))
                collision_reward = Config.hit_penalty * (scaling + 0.1)
                if collision_reward >= 700:
                    terminal = True
                reward -= collision_reward

        reward -= pow(goal_dist / 4935.0, 0.8) * 1.2
        # reward -= pow(goal_dist / 3000.0, 0.8) * 1.2

        # TODO: Replace the below with all grid positions of incoming_car in player rectangle
        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            x = self.world.incoming_car.get_location().x
            y = self.world.incoming_car.get_location().y
            grid[round(x), round(y)] = 100

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
            if action != 0 and self.prev_speed < 0.5:
                reward -= Config.braking_penalty
            # if action == 0 and self.prev_speed < 0.5:
            #     reward += Config.braking_penalty

        # Limit max speed to 45
        if self.prev_speed is not None:
            if action == 0 and self.prev_speed > 45:
                reward -= Config.braking_penalty

        # Penalize braking/acceleration actions to get a smoother ride
        if self.prev_action.brake > 0: last_action = 2
        elif self.prev_action.throttle > 0: last_action = 0
        else: last_action = 1
        if last_action != 0 and last_action != action:
            reward -= 0.05
        # if action == 0:
        #     reward -= 0.05
        # if action == 2:
        #     reward -= 0.05

        reward -= pow(abs(self.prev_action.steer), 1.3) / 2.0

        if goal_dist < 3:
            reward += Config.goal_reward
            goal = True
            terminal = True

        # incentive breaking if pedestrian nearby
        # if self.in_near_miss(start[0], start[1], start[2], walker_x, walker_y, front_margin=5, side_margin=1.0):
        #     if action == 2:
        #         reward += 100

        # Normalize reward
        reward = reward / 1000.0

        # hit = hit or self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                                front_margin=0, side_margin=0, back_margin=0)
        hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                front_margin=0.2, side_margin=0.2, back_margin=0.1) or hit
        nearmiss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=1.5, side_margin=0.5, back_margin=0.5)
        return reward, goal, hit, nearmiss, terminal

    def get_reward_hybrid(self):
        reward = 0
        goal = False
        hit = False
        nearmiss = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5)
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if speed > 0.3:
            walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
            ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                        front_margin=2, side_margin=1.2)

            if ped_hit:
                ped_collision_reward = -0.2 + (-1 * pow(0.5 + speed / Config.max_speed, 1.4))
                hit = True
                nearmiss = True
                reward += ped_collision_reward

        if goal_dist < 3:
            reward += 1
            goal = True

        if self.prev_action.throttle != 0:
            reward -= 0.01

        reward -= 0.5 * abs(speed - Config.max_speed) / 10000

        return reward, goal, hit, nearmiss

    def get_reward_old(self):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        goal = False
        near_miss = False
        hit = False
        if goal_dist < 3:
            goal = True
            return 1000, goal, hit, near_miss

        # Pedestrian hit and near miss section
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        # In hit area
        hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                front_margin=0, side_margin=0, back_margin=0)
        if hit:  # accident
            hit = True
            return -1000, goal, hit, near_miss
        # in near miss area
        near_miss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                      front_margin=1.5, side_margin=0.5)

        # TODO: Collision with incoming or static car

        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            x = self.world.incoming_car.get_location().x
            y = self.world.incoming_car.get_location().y
            grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        location = [min(round(start[0] - self.min_x), self.grid_cost.shape[0] - 1),
                    min(round(start[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        reward = -grid[location[0], location[1]]

        reward += -0.1
        if self.prev_action.throttle != 0:
            reward += -0.1
        if self.prev_action.steer != 0:
            reward += -1
        if near_miss:  # near miss
            reward += -500
        return reward, goal, hit, near_miss

    def get_local_coordinates(self, path):
        world_to_camera = np.array(self.world.semseg_sensor.sensor.get_transform().get_inverse_matrix())
        world_points = np.array(path)
        world_points = world_points[:, :2]
        world_points = np.c_[world_points, np.zeros(world_points.shape[0]), np.ones(world_points.shape[0])].T
        sensor_points = np.dot(world_to_camera, world_points)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        image_w = int(Config.segcam_image_x)
        image_h = int(Config.segcam_image_y)
        fov = float(Config.segcam_fov)
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0
        points_2d = np.dot(K, point_in_camera_coords)
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])
        points_2d = points_2d.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)
        return u_coord, v_coord

    def get_car_intention(self, obstacles, path, start):
        self.past_trajectory.append(start)
        car_intention = self.world.semseg_sensor.array.copy()
        if len(path) == 0:
            return car_intention
        x, y = self.get_local_coordinates(path)
        car_intention[y, x, :] = 255.0  # overlay planned path on input with white line

        x, y = self.get_local_coordinates(self.past_trajectory)
        car_intention[y, x, :] = 0.0  # overlay past trajectory on input with black line
        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(car_intention, file)
        # car_intention = np.transpose(car_intention, (2, 0, 1))
        # assert car_intention.shape[0] == 3
        return car_intention

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 50.0:
            if self.scenario[0] == 3 and walker_x >= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
            elif self.scenario[0] in [7, 8] and walker_x <= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
            elif self.scenario[0] in [1, 2, 4, 5, 6, 9, 10]:
                obstacles.append((int(walker_x), int(walker_y)))
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                obstacles.append((int(car_x), int(car_y)))
        t0 = time.time()
        paths = self.path_planner.find_path(start, end, self.grid_cost, obstacles)
        if len(paths):
            path = paths[0]
        else:
            path = []
        # print("Time taken to generate path {:.4f}ms".format((time.time() - t0) * 1000))
        path.reverse()

        if self.display_costmap:
            self.plot_costmap(obstacles, path)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            control.steer = (path[2][2] - start[2]) / 70.

        self.prev_action = control
        velocity = self.vehicle.get_velocity()
        self.prev_speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5)
        return control, self.get_car_intention(obstacles, path, start)

    def plot_costmap(self, obstacles, path):
        cp = self.occupancy_grid.get_costmap([])
        x, y = list(), list()
        for node in path:
            pixel_coord = self.occupancy_grid.map.convert_to_pixel(node)
            x.append(pixel_coord[0])
            y.append(pixel_coord[1])
        plt.plot(x, y, "-r")
        if len(obstacles) > 0:
            obstacle_pixel = self.occupancy_grid.map.convert_to_pixel([obstacles[0][0], obstacles[0][1], 0])
            plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        # print(obstacle_pixel)
        plt.imshow(cp, cmap="gray")
        plt.axis([0, 350, 950, 1550])
        plt.draw()
        plt.pause(0.1)
        self.fig.clear()
