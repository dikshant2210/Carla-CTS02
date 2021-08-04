"""
Author: Dikshant Gupta
Time: 02.08.21 13:24
"""

import datetime
import carla
from agents.navigation.config import Config
import cv2


class SegmentationSensor:
    def __init__(self, world, player):
        self.config = Config
        self.bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.bp.set_attribute('image_size_x', self.config.occupancy_grid_width)
        self.bp.set_attribute('image_size_y', self.config.occupancy_grid_height)
        self.bp.set_attribute('fov', self.config.segcam_fov)
        self.bp.set_attribute('sensor_tick', self.config.sensor_simulation_step)

        location = player.get_location()
        transform = carla.Transform(carla.Location(x=location.x, z=60), carla.Rotation(pitch=-90))
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=player)
        self.sensor.listen(lambda image: self.save_img(image))

    @staticmethod
    def save_img(image):
        ts = datetime.datetime.now().timestamp()
        cc = carla.ColorConverter()
        image.save_to_disk("_out/segmap_{}.jpg".format(ts), cc.CityScapesPalette)
        # img = cv2.imread("_out/segmap_{}.jpg".format(ts))
        # print(img.shape)
        # cv2.imshow('segmap', img)
