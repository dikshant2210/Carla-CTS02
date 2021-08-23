"""
Author: Dikshant Gupta
Time: 21.08.21 10:08
"""

import random
import carla


class Scenario:
    def __init__(self, world):
        self.world = world

    def scenario10(self):
        end = (2, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 5.2
        walker_spawn_point.location.y = 235.0 - 45
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = -1.5
        car_spawn_point.location.y = 150
        car_spawn_point.location.z = 0.01
        car_spawn_point.rotation.yaw = 90
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        return 10, obstacles, end
