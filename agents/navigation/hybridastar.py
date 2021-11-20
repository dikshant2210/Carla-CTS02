"""
Author: Dikshant Gupta
Time: 16.08.21 23:55
"""

# It finds the optimal path for a car using Hybrid A* and bicycle model.

import heapq as hq
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from assets.occupancy_grid import OccupancyGrid
import pickle as pkl


# total cost f(n) = actual cost g(n) + heuristic cost h(n)
class HybridAStar:
    def __init__(self, min_x, max_x, min_y, max_y, obstacle=(), vehicle_length=2):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.obstacle = obstacle
        self.vehicle_length = vehicle_length
        print("Vehicle length: {:.2f}".format(vehicle_length))

        self.obstacles = set(self.obstacle)

    def hgcost(self, position, target, occupancy_grid):
        # Euclidean distance
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) + (
                math.radians(position[2]) - math.radians(target[2])) ** 2)
        x = round(position[0] - self.min_x)
        location = [min(x, occupancy_grid.shape[0] - 1),
                    min(round(position[1] - self.min_y), occupancy_grid.shape[1] - 1)]
        cost = occupancy_grid[location[0], location[1]]
        return float(output + cost)

    def dist(self, position, target):
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) +
                         (math.radians(position[2]) - math.radians(target[2])) ** 2)
        return float(output)

    def next_node(self, location, aph, d):
        theta = math.radians(location[2])
        alpha = math.radians(aph)
        if aph < 0.0001:
            new_x = location[0] + d * math.cos(theta)
            new_y = location[1] + d * math.sin(theta)
            new_theta = theta + (d * math.tan(alpha) / self.vehicle_length)
        else:
            beta = d * math.tan(alpha) / self.vehicle_length
            radii = d / beta
            center_x = location[0] - math.sin(theta) * radii
            center_y = location[1] + math.cos(theta) * radii

            new_x = center_x + math.sin(theta + beta) * radii
            new_y = center_y - math.cos(theta + beta) * radii
            new_theta = theta + beta

        return new_x, new_y, new_theta

    def find_path(self, start, end, occupancy_grid, agent_locations, m=1):
        # steering_inputs = [-50, 0, 50]
        # cost_steering_inputs = [0.1, 0, 0.1]
        steering_inputs = []
        for i in range(-50, 51, 25):
            steering_inputs.append(i)
        # print(steering_inputs)
        # print(occupancy_grid.shape)

        speed_inputs = [1.05]
        # cost_speed_inputs = [0]

        start = (float(start[0]), float(start[1]), float(start[2]))
        end = (float(end[0]), float(end[1]), float(end[2]))
        # The above 2 are in discrete coordinates

        open_heap = []  # element of this list is like (cost,node_d)
        open_diction = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))

        visited_diction = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))

        obstacles = agent_locations
        cost_to_neighbour_from_start = 0

        heuristic_cost = self.hgcost(start, end, occupancy_grid)
        hq.heappush(open_heap, (cost_to_neighbour_from_start + heuristic_cost, start))

        open_diction[start] = (cost_to_neighbour_from_start + heuristic_cost, start, (start, start))

        paths = list()
        while len(open_heap) > 0:
            while True:
                chosen_d_node = open_heap[0][1]
                if chosen_d_node in visited_diction:
                    hq.heappop(open_heap)
                else:
                    break
            chosen_node_total_cost = open_heap[0][0]
            chosen_c_node = open_diction[chosen_d_node][1]

            visited_diction[chosen_d_node] = open_diction[chosen_d_node]

            if self.dist(chosen_d_node, end) < 1:
                m -= 1  # Found one path to goal
                rev_final_path = [end]  # reverse of final path
                rev_final_path_d = [end]  # reverse of discrete final path
                node = chosen_d_node
                while True:
                    # visited_diction
                    open_node_contents = visited_diction[node]  # (cost,node_c,(parent_d,parent_c))
                    parent_of_node = open_node_contents[2][1]

                    rev_final_path.append(parent_of_node)
                    rev_final_path_d.append(open_node_contents[2][0])
                    node = open_node_contents[2][0]
                    if node == start:
                        rev_final_path.append(start)
                        break
                paths.append(rev_final_path)
                if m == 0:
                    return paths
                open_heap = []
                visited_diction = {}
                heuristic_cost = self.hgcost(start, end, occupancy_grid)
                hq.heappush(open_heap, (cost_to_neighbour_from_start + heuristic_cost, start))
                for p in rev_final_path_d[:-1]:
                    visited_diction[p] = open_diction[p]
                open_diction = dict()
                open_diction[start] = (heuristic_cost, start, (start, start))
                continue

            hq.heappop(open_heap)
            for i in range(len(steering_inputs)):
                for j in range(len(speed_inputs)):

                    delta = steering_inputs[i]
                    velocity = speed_inputs[j]

                    cost_to_neighbour_from_start = chosen_node_total_cost - self.hgcost(chosen_d_node, end,
                                                                                        occupancy_grid)
                    neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts = self.next_node(chosen_c_node,
                                                                                           delta, velocity)
                    neighbour_theta_cts = math.degrees(neighbour_theta_cts)

                    neighbour_x_d = round(neighbour_x_cts)
                    neighbour_y_d = round(neighbour_y_cts)
                    neighbour_theta_d = round(neighbour_theta_cts)

                    neighbour = ((neighbour_x_d, neighbour_y_d, neighbour_theta_d),
                                 (neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts))

                    dist = 1000
                    for obs in obstacles:
                        d = np.sqrt((neighbour_x_d - obs[0]) ** 2 + (neighbour_y_d - obs[1]) ** 2)
                        if d < dist:
                            dist = d
                    if ((dist > 1.5) and (neighbour_x_d >= self.min_x)
                            and (neighbour_x_d <= self.max_x) and (neighbour_y_d >= self.min_y) and
                            (neighbour_y_d <= self.max_y)):

                        heurestic = self.hgcost((neighbour_x_d, neighbour_y_d, neighbour_theta_d), end,
                                                occupancy_grid)
                        # adding the unit action cost and distance travelled
                        action_cost = 1.0
                        cost_to_neighbour_from_start = action_cost + cost_to_neighbour_from_start

                        total_cost = heurestic + cost_to_neighbour_from_start

                        skip = 0

                        if (neighbour[0] in visited_diction) and (total_cost < visited_diction[neighbour[0]][0]):
                            visited_diction[neighbour[0]] = (total_cost, neighbour[1], (chosen_d_node, chosen_c_node))
                            skip = 1

                        if (neighbour[0] in open_diction) and (total_cost > open_diction[neighbour[0]][0]):
                            skip = 1

                        if skip == 0:
                            hq.heappush(open_heap, (total_cost, neighbour[0]))
                            open_diction[neighbour[0]] = (total_cost, neighbour[1], (chosen_d_node, chosen_c_node))
        # print("Did not find the goal - it's unattainable.")
        return []


def main():
    print(__file__ + " start!!")

    # start and goal position
    # (x, y, theta) in meters, meters, degrees
    sx, sy, stheta = 92, 160, -90
    gx, gy, gtheta = 75, 1, -180

    # sx, sy, stheta = 100, 1, -180
    # gx, gy, gtheta = 70, 1, -180

    # create obstacles
    obstacle = [(1.4, 200), (-1.5, 190)]  # , (2, 231), (1, 230), (1, 231), (3, 230), (3, 231)]
    occupancy_grid = OccupancyGrid()

    g = np.ones((110, 310)) * 1000.0
    g[7:13, 13:] = 1.0
    g[97:103, 13:] = 1.0
    g[7:, 7:13] = 1.0
    # plt.imshow(g.T, cmap='gray')
    # plt.show()
    hy_a_star = HybridAStar(-10, 100, -10, 300, obstacle=[], vehicle_length=4)
    print(hy_a_star.hgcost((sx, sy, stheta), (gx, gy, gtheta), g))
    t0 = time.time()
    paths = hy_a_star.find_path((sx, sy, stheta), (gx, gy, gtheta), g, obstacle)
    if paths:
        path = paths[0]
    else:
        path = []
    print("Time taken: {:.4f}ms".format((time.time() - t0) * 1000))
    path.reverse()
    print(len(path))

    cp = occupancy_grid.get_costmap([])
    for path in paths:
        x, y = list(), list()
        for node in path:
            pixel_coord = occupancy_grid.map.convert_to_pixel(node)
            x.append(pixel_coord[0])
            y.append(pixel_coord[1])
        plt.plot(x, y, "-r")
        obstacle_pixel = occupancy_grid.map.convert_to_pixel([obstacle[0][0], obstacle[0][1], 0])
        plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        obstacle_pixel = occupancy_grid.map.convert_to_pixel([obstacle[1][0], obstacle[1][1], 0])
        plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        plt.imshow(cp, cmap='gray')
        # plt.imshow(cp[x[0]-50:x[0]+50, y[0]-200:y[0]+500], cmap="gray")
        plt.show()

    # k = 1
    # for _ in range(k):
    #     for node in path[5:-5]:
    #         obstacle.append((int(node[0]), int(node[1])))
    #         # ox.append(int(node[0]))
    #         # oy.append(int(node[1]))
    #     print(len(obstacle))
    #     t0 = time.time()
    #     paths = hy_a_star.find_path((sx, sy, stheta), (gx, gy, gtheta), grid_cost, obstacle)
    #     print("Time taken: {:.4f}ms".format((time.time() - t0) * 1000))
    #     for path in paths:
    #         x, y = list(), list()
    #         for node in path:
    #             pixel_coord = occupancy_grid.map.convert_to_pixel(node)
    #             x.append(pixel_coord[0] + 340)
    #             y.append(pixel_coord[1])
    #         plt.plot(x, y, "-b")


if __name__ == '__main__':
    main()
