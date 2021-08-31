"""
Author: Dikshant Gupta
Time: 16.08.21 23:55
"""

# It finds the optimal path for a car using Hybrid A* and bicycle model.

import heapq as hq
import math
import numpy as np


# total cost f(n) = actual cost g(n) + heuristic cost h(n)
class HybridAStar:
    def __init__(self, min_x, max_x, min_y, max_y, obstacle=(), resolution=1, vehicle_length=2):
        # TODO

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.obstacle = obstacle
        self.resolution = resolution
        self.vehicle_length = vehicle_length

        self.obstacles = set(self.obstacle)

        ###

    def euc_dist(self, position, target, occupancy_grid):
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) + (
                math.radians(position[2]) - math.radians(target[2])) ** 2)
        grid_location = occupancy_grid.map.convert_to_pixel(position)
        # print(float(output), occupancy_grid.static_map[grid_location[0] + 340, grid_location[1]])
        return float(output) + (occupancy_grid.static_map[grid_location[0] + 340, grid_location[1]] / 10.0)

    """
    For each node n, we need to store:
    (discrete_x, discrete_y, heading angle theta),
    (continuous x, continuous y, heading angle theta)
    cost g, f,
    path [(continuous x, continuous y, continuous theta),...]

    start: discrete (x, y, theta)
    end: discrete (x, y, theta)
    sol_path = [(x1,y1,theta1),(x2,y2,theta2), ...]
    """

    def find_path(self, start, end, occupancy_grid, agent_locations):
        steering_inputs = []
        cost_steering_inputs = []
        for i in range(-25, 26, 25):
            steering_inputs.append(i)
            if i == 0:
                cost_steering_inputs.append(0)
            else:
                cost_steering_inputs.append(0.1)

        speed_inputs = [1]
        cost_speed_inputs = [0]

        start = (float(start[0]), float(start[1]), float(start[2]))
        end = (float(end[0]), float(end[1]), float(end[2]))
        # The above 2 are in discrete coordinates

        open_heap = []  # element of this list is like (cost,node_d)
        open_diction = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))

        visited_diction = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))

        obstacles = set(self.obstacle + agent_locations)
        print(obstacles)
        cost_to_neighbour_from_start = 0

        # Here a heap is chosen. (cost, path) is a tuple which is pushed in
        # open_set_sorted. The main advantage is that as more (cost, path) are
        # added to this open_set, heap automatically sorts it and this first
        # element is automatically the lowest cost one
        # Here path is [(),()....] where each () has (discrete,continuous) for a node
        # for path normal appending is done. If you use heap there, the elements
        # get sorted and we don't want that. We want to preserve the order in
        # which we move for start to destination node

        heuristic_cost = self.euc_dist(start, end, occupancy_grid)
        hq.heappush(open_heap, (cost_to_neighbour_from_start + heuristic_cost, start))

        open_diction[start] = (cost_to_neighbour_from_start + heuristic_cost, start, (start, start))

        while len(open_heap) > 0:

            # choose the node that has minimum total cost for exploration
            # print(len(open_heap))

            chosen_d_node = open_heap[0][1]
            chosen_node_total_cost = open_heap[0][0]
            chosen_c_node = open_diction[chosen_d_node][1]

            visited_diction[chosen_d_node] = open_diction[chosen_d_node]

            # print(self.euc_dist(chosen_path_last_element[0],end))

            if self.euc_dist(chosen_d_node, end, occupancy_grid) < 1:

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
                final_path = []
                for p in rev_final_path:
                    final_path.append(p)
                return final_path

            hq.heappop(open_heap)

            for i in range(0, 3):
                for j in range(0, 1):

                    delta = steering_inputs[i]
                    velocity = speed_inputs[j]

                    cost_to_neighbour_from_start = chosen_node_total_cost - self.euc_dist(chosen_d_node, end, occupancy_grid)

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

                    if (((neighbour_x_d, neighbour_y_d) not in obstacles) and (neighbour_x_d >= self.min_x)
                            and (neighbour_x_d <= self.max_x) and (neighbour_y_d >= self.min_y) and
                            (neighbour_y_d <= self.max_y)):

                        heurestic = self.euc_dist((neighbour_x_d, neighbour_y_d, neighbour_theta_d), end, occupancy_grid)
                        cost_to_neighbour_from_start = abs(velocity) + cost_to_neighbour_from_start + \
                                                       cost_steering_inputs[i] + cost_speed_inputs[j]

                        # print(heuristic,cost_to_neighbour_from_start)
                        total_cost = heurestic + cost_to_neighbour_from_start

                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor

                        skip = 0
                        # print(open_set_sorted)
                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor
                        found_lower_cost_path_in_open = 0

                        if neighbour[0] in open_diction:

                            if total_cost > open_diction[neighbour[0]][0]:
                                skip = 1

                            elif neighbour[0] in visited_diction:

                                if total_cost > visited_diction[neighbour[0]][0]:
                                    found_lower_cost_path_in_open = 1

                        if skip == 0 and found_lower_cost_path_in_open == 0:
                            hq.heappush(open_heap, (total_cost, neighbour[0]))
                            open_diction[neighbour[0]] = (total_cost, neighbour[1], (chosen_d_node, chosen_c_node))
            # a=a+1
            # print(open_set_sorted)
        print("Did not find the goal - it's unattainable.")
        return []
