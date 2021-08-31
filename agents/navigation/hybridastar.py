"""
Author: Dikshant Gupta
Time: 16.08.21 23:55
"""

# It finds the optimal path for a car using Hybrid A* and bicycle model.

import heapq as hq
import math
import carla
import matplotlib.pyplot as plt
import numpy as np
import time
from assets.occupancy_grid import OccupancyGrid


# total cost f(n) = actual cost g(n) + heuristic cost h(n)
class HybridAStar:
    def __init__(self, min_x, max_x, min_y, max_y, obstacle=(), vehicle_length=2):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.obstacle = obstacle
        self.vehicle_length = vehicle_length

        self.obstacles = set(self.obstacle)

    def hgcost(self, position, target, occupancy_grid):
        # Euclidean distance
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) + (
                math.radians(position[2]) - math.radians(target[2])) ** 2)

        cost = occupancy_grid[round(position[0] - self.min_x), round(position[1] - self.min_y)]
        return float(output + cost)

    def dist(self, position, target):
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) +
                         (math.radians(position[2]) - math.radians(target[2])) ** 2)
        return float(output)

    def find_path(self, start, end, occupancy_grid, agent_locations):
        # steering_inputs = [-50, 0, 50]
        # cost_steering_inputs = [0.1, 0, 0.1]
        steering_inputs = []
        cost_steering_inputs = [0.2, 0.1, 0, 0.1, 0.2]
        for i in range(-50, 51, 50):
            steering_inputs.append(i)
        # print(steering_inputs, cost_steering_inputs)

        speed_inputs = [1.05]
        cost_speed_inputs = [0]

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

        while len(open_heap) > 0:
            while True:
                chosen_d_node = open_heap[0][1]
                if chosen_d_node in visited_diction:
                    hq.heappop(open_heap)
                else:
                    break
            # chosen_d_node = open_heap[0][1]
            chosen_node_total_cost = open_heap[0][0]
            chosen_c_node = open_diction[chosen_d_node][1]

            visited_diction[chosen_d_node] = open_diction[chosen_d_node]

            if self.dist(chosen_d_node, end) < 1:

                rev_final_path = [end]  # reverse of final path
                node = chosen_d_node
                m = 1
                while m == 1:
                    # visited_diction
                    open_node_contents = visited_diction[node]  # (cost,node_c,(parent_d,parent_c))
                    parent_of_node = open_node_contents[2][1]

                    rev_final_path.append(parent_of_node)
                    node = open_node_contents[2][0]
                    if node == start:
                        rev_final_path.append(start)
                        break
                return rev_final_path

            hq.heappop(open_heap)

            for i in range(len(steering_inputs)):
                for j in range(len(speed_inputs)):

                    delta = steering_inputs[i]
                    velocity = speed_inputs[j]

                    cost_to_neighbour_from_start = chosen_node_total_cost - self.hgcost(chosen_d_node, end,
                                                                                        occupancy_grid)

                    neighbour_x_cts = chosen_c_node[0] + (velocity * math.cos(math.radians(chosen_c_node[2])))
                    neighbour_y_cts = chosen_c_node[1] + (velocity * math.sin(math.radians(chosen_c_node[2])))
                    neighbour_theta_cts = math.radians(chosen_c_node[2]) + (
                            velocity * math.tan(math.radians(delta)) / (float(self.vehicle_length)))

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

                        heurestic = self.hgcost((neighbour_x_d, neighbour_y_d, neighbour_theta_d), end, occupancy_grid)
                        # cost_to_neighbour_from_start = abs(velocity) + cost_to_neighbour_from_start

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
        print("Did not find the goal - it's unattainable.")
        return []


def main():
    print(__file__ + " start!!")

    # start and goal position
    # (x, y, theta) in meters, meters, degrees
    sx, sy, stheta = 2, 250, -90
    gx, gy, gtheta = 2, 180, -90  # 2,4,0 almost exact

    # create obstacles
    obstacle = [(2.5, 200), (0, 198)]  # , (2, 231), (1, 230), (1, 231), (3, 230), (3, 231)]

    hy_a_star = HybridAStar(-10, 396, -10, 330, obstacle=[], vehicle_length=2)
    occupancy_grid = OccupancyGrid()
    sloc = occupancy_grid.map.convert_to_pixel([sx, sy, stheta])
    gloc = occupancy_grid.map.convert_to_pixel([gx, gy, gtheta])
    print(occupancy_grid.static_map[sloc[0] + 340, sloc[1]])
    print(occupancy_grid.static_map[gloc[0] + 340, gloc[1]])

    grid_cost = np.ones((396 + 10, 330 + 10))
    print(occupancy_grid.static_map.shape)
    for i in range(396 + 10):
        for j in range(330 + 10):
            loc = occupancy_grid.map.convert_to_pixel([i, j, 0])
            x = loc[0] + 340
            if x > 2199:
                x = 2199
            val = occupancy_grid.static_map[x, loc[1]]
            grid_cost[i, j] = val
    print(np.unique(grid_cost, return_counts=True))
    t0 = time.time()
    path = hy_a_star.find_path((sx, sy, stheta), (gx, gy, gtheta), grid_cost, obstacle)
    print("Time taken: {:.4f}ms".format((time.time() - t0) * 1000))

    cp = occupancy_grid.get_costmap([])
    x, y = list(), list()
    for node in path:
        pixel_coord = occupancy_grid.map.convert_to_pixel(node)
        x.append(pixel_coord[0] + 340)
        y.append(pixel_coord[1])
    plt.plot(x, y, "-r")
    obstacle_pixel = occupancy_grid.map.convert_to_pixel([obstacle[0][0], obstacle[0][1], 0])
    plt.scatter([obstacle_pixel[0] + 340], [obstacle_pixel[1]], c="k")
    obstacle_pixel = occupancy_grid.map.convert_to_pixel([obstacle[1][0], obstacle[1][1], 0])
    plt.scatter([obstacle_pixel[0] + 340], [obstacle_pixel[1]], c="k")
    plt.imshow(cp)

    # k = 1
    # for _ in range(k):
    #     for node in path[5:-5]:
    #         obstacle.append((int(node[0]), int(node[1])))
    #         # ox.append(int(node[0]))
    #         # oy.append(int(node[1]))
    #     print(len(obstacle))
    #     t0 = time.time()
    #     path = hy_a_star.find_path((sx, sy, stheta), (gx, gy, gtheta), grid_cost, obstacle)
    #     print("Time taken: {:.4f}ms".format((time.time() - t0) * 1000))
    #     rx1, ry1 = [], []
    #     for node in path:
    #         rx1.append(node[0])
    #         ry1.append(node[1])
    #     plt.plot(rx1, ry1, "-b")

    plt.show()


if __name__ == '__main__':
    main()
