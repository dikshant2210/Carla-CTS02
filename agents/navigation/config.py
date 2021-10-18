"""
Author: Dikshant Gupta
Time: 25.07.21 09:57
"""


class Config:
    PI = 3.14159

    simulation_step = 0.1  # 0.008
    sensor_simulation_step = '0.5'
    synchronous = True

    grid_size = 2  # grid size in meters
    speed_limit = 50
    max_steering_angle = 1.22173  # 70 degrees in radians
    occupancy_grid_width = '1920'
    occupancy_grid_height = '1080'

    location_threshold = 1.0

    ped_speed_range = [0.6, 3.3]
    ped_distance_range = [0, 40]
    car_speed_range = [6, 9]
    scenarios = ['01']
