"""
Author: Dikshant Gupta
Time: 03.08.21 13:22
"""

import numpy as np
import matplotlib.pyplot as plt
from assets.map import CarlaMap


class OccupancyGrid:
    def __init__(self, map_name="Town01"):
        self.map = CarlaMap(map_name)
        self.ref = self.map.get_map()
        # extract green channel, invert, scale to range 0..100, convert to int8
        self.ref = (self.ref[..., 1] * 100.0 / 255).astype(np.int8)

        self.static_map = np.zeros(self.ref.shape)
        self.static_map[:, :] = 10000
        self.static_map[self.ref == 100] = 1  # road
        self.static_map[(self.ref > 94) & (self.ref < 100)] = 50  # sidewalk
        self.static_map[self.ref == 70] = 100  # obstacle

    def get_costmap(self, agents):
        costmap = np.copy(self.static_map)
        for agent in agents:
            pass
        return self.ref


if __name__ == '__main__':
    grid = OccupancyGrid("Town01")
    cp = grid.get_costmap([])
    cp[cp == 10000] = -56
    plt.imshow(cp, cmap="gray")
    plt.show()
