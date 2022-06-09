"""
Author: Dikshant Gupta
Time: 10.11.21 01:14
"""

import carla
from hyleap.connectors import train_connector, image_connector, ConnectorServer
from agents.navigation.hylear_controller import HyLEAR


class HyLEAP(HyLEAR):
    def __init__(self, world, carla_map, scenario, conn=None):
        super(HyLEAP, self).__init__(world, carla_map, scenario, conn, eval_mode=False, agent="hyleap")
        self.train_connection = train_connector()
        self.image_connection = image_connector()
        self.connection = ConnectorServer(0)
        self.train_connection.start()
        self.image_connection.start()
        self.connection.start()
        # self.connection.join()

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

        # Best speed action for the given path
        if not self.eval_mode:
            control = self.get_speed_action(path, control)
        self.prev_action = control
        return control, intention, risk
