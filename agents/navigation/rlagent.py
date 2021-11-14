"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import sys
import datetime
import time
import numpy as np
import pickle as pkl
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
        self.folder = datetime.datetime.now().timestamp()
        self.past_trajectory = []
        # os.mkdir("_out/{}".format(self.folder))

        wps = carla_map.generate_waypoints(Config.grid_size)
        print("Total no. of waypoints: {}".format(len(wps)))

        self.max_x = -sys.maxsize - 1
        self.max_y = -sys.maxsize - 1
        self.min_x = sys.maxsize
        self.min_y = sys.maxsize

        for wp in wps:
            xyz = wp.transform.location
            self.max_x = max([self.max_x, xyz.x])
            self.max_y = max([self.max_y, xyz.y])
            self.min_x = min([self.min_x, xyz.x])
            self.min_y = min([self.min_y, xyz.y])

        # print(self.max_y, self.max_x, self.min_y, self.min_x)

        self.max_x = int(self.max_x)
        self.max_y = int(self.max_y)
        self.min_y = int(self.min_y) - 10
        self.min_x = int(self.min_x) - 10

        obstacle = []
        self.vehicle_length = self.vehicle.bounding_box.extent.x * 2
        self.vehicle_width = self.vehicle.bounding_box.extent.y * 2
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y,
                                        obstacle, self.vehicle_length)

        x_range = self.max_x - self.min_x
        y_range = self.max_y - self.min_y
        self.grid_cost = np.ones((x_range, y_range))
        for i in range(self.min_x, self.max_x):
            for j in range(self.min_y, self.max_y):
                loc = self.occupancy_grid.map.convert_to_pixel([i, j, 0])
                x = loc[0]
                y = loc[1]
                if x > 2199:
                    x = 2199
                if y > 2599:
                    y = 2599
                val = self.occupancy_grid.static_map[x, y]
                if val == 50:
                    val = 50.0
                self.grid_cost[i - self.min_x, j - self.min_y] = val

    def update_scenario(self, scenario):
        self.scenario = scenario
        self.past_trajectory = []

    def in_hit_area(self, x, y, theta, ped_x, ped_y):
        # TOP RIGHT VERTEX:
        top_right_x = x + ((self.vehicle_width / 2) * np.cos(theta)) - ((self.vehicle_length / 2) * np.sin(theta))
        top_right_y = y + ((self.vehicle_width / 2) * np.sin(theta)) + ((self.vehicle_length / 2) * np.cos(theta))

        # TOP LEFT VERTEX:
        top_left_x = x - ((self.vehicle_width / 2) * np.cos(theta)) - ((self.vehicle_length / 2) * np.sin(theta))
        top_left_y = y - ((self.vehicle_width / 2) * np.sin(theta)) + ((self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x - ((self.vehicle_width / 2) * np.cos(theta)) + ((self.vehicle_length / 2) * np.sin(theta))
        bot_left_y = y - ((self.vehicle_width / 2) * np.sin(theta)) - ((self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x + ((self.vehicle_width / 2) * np.cos(theta)) + ((self.vehicle_length / 2) * np.sin(theta))
        bot_right_y = y + ((self.vehicle_width / 2) * np.sin(theta)) - ((self.vehicle_length / 2) * np.cos(theta))

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]
        return 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

    def in_near_miss(self, x, y, theta, ped_x, ped_y, front_margin=1.5, side_margin=0.5):
        # TOP RIGHT VERTEX:
        top_right_x = x + ((side_margin + self.vehicle_width / 2) * np.cos(theta)) - \
                      ((front_margin + self.vehicle_length / 2) * np.sin(theta))
        top_right_y = y + ((side_margin + self.vehicle_width / 2) * np.sin(theta)) + \
                      ((front_margin + self.vehicle_length / 2) * np.cos(theta))

        # TOP LEFT VERTEX:
        top_left_x = x - ((side_margin + self.vehicle_width / 2) * np.cos(theta)) - \
                     ((front_margin + self.vehicle_length / 2) * np.sin(theta))
        top_left_y = y - ((side_margin + self.vehicle_width / 2) * np.sin(theta)) + \
                     ((front_margin + self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x - ((side_margin + self.vehicle_width / 2) * np.cos(theta)) + \
                     ((0.5 + self.vehicle_length / 2) * np.sin(theta))
        bot_left_y = y - ((side_margin + self.vehicle_width / 2) * np.sin(theta)) - \
                     ((0.5 + self.vehicle_length / 2) * np.cos(theta))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x + ((side_margin + self.vehicle_width / 2) * np.cos(theta)) + \
                      ((0.5 + self.vehicle_length / 2) * np.sin(theta))
        bot_right_y = y + ((side_margin + self.vehicle_width / 2) * np.sin(theta)) - \
                      ((0.5 + self.vehicle_length / 2) * np.cos(theta))

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]
        return 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

    def get_reward(self):
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
            ped_hit = self.in_near_miss(start[0], start[1], start[2], walker_x, walker_y,
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
        hit = self.in_hit_area(start[0], start[1], start[2], walker_x, walker_y)
        if hit:  # accident
            hit = True
            return -1000, goal, hit, near_miss
        # in near miss area
        near_miss = self.in_near_miss(start[0], start[1], start[2], walker_x, walker_y)

        # TODO: Collision with incoming or static car

        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            x = self.world.incoming_car.get_location().x
            y = self.world.incoming_car.get_location().y
            grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        if self.scenario[0] in [1, 2, 3, 6, 9, 10]:
            x = round(start[0] - self.min_x - 1)
        else:
            x = round(start[0] - self.min_x)
        location = [min(x, self.grid_cost.shape[0] - 1),
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
