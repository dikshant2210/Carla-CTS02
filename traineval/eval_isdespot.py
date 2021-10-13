"""
Author: Dikshant Gupta
Time: 13.09.21 16:37
"""
import carla
import pygame
import argparse
import logging
import time
import subprocess
import random
from multiprocessing import Process
import numpy as np
from world import World
from hud import HUD
from traineval.baseagent import BaseAgent
from traineval.traineval_utils import KeyboardControl
from agents.tools.connector import Connector
from agents.navigation.config import Config
from agents.tools.scenario import Scenario


class ISDespot(BaseAgent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, conn, scenario):
        super(ISDespot, self).__init__(world, carla_map, conn, scenario)

    def run_step(self, debug=False):
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        if self.prev_action is not None:
            reward, terminal = self.get_reward()
        else:
            # handling first instance
            reward = 0
            terminal = False
        if terminal:
            return "goal"
        angle = transform.rotation.yaw
        car_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        pedestrian_positions = [[self.world.walker.get_location().x, self.world.walker.get_location().y]]

        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 50.0:
            obstacles.append((int(walker_x), int(walker_y)))
            pedestrian_positions = [[walker_x, walker_y]]
        if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
            obstacles.append((int(car_x), int(car_y)))
        paths = self.path_planner.find_path(start, end, self.grid_cost, obstacles)
        if len(paths):
            path = paths[0]
        else:
            path = []
        path.reverse()

        if self.display_costmap:
            self.plot_costmap(obstacles, path)

        control = carla.VehicleControl()
        if len(path) == 0:
            control.throttle = -1
            control.steer = 0
        else:
            t0 = time.time()
            self.conn.send_message(terminal, reward, angle, car_pos, car_speed, pedestrian_positions, path)
            m = self.conn.receive_message()
            acc = 0
            if m[0] == '0':
                acc = 0.6
            elif m[0] == '2':
                acc = -0.6
            control.throttle = acc
            control.steer = (path[2][2] - start[2]) / 70.
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        self.prev_action = control
        return control


class RunISDespot:
    def __init__(self, speed_range, distance_range, args, scenarios=9):
        self.exp_setting = list()
        self.args = args

        # Get all combinations of scenarios and speed-distance trial combinations
        for scenario in range(1, scenarios + 1):
            for speed in range(speed_range[0], speed_range[1] + 1):
                for distance in range(distance_range[0], distance_range[1] + 1):
                    self.exp_setting.append((scenario, speed, distance))

        random.shuffle(self.exp_setting)

        # Setting up the simulator
        pygame.init()
        pygame.font.init()
        self.despot_port = 1245

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(2.0)

        if args.display:
            self.display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0, 0, 0))
            pygame.display.flip()

        self.hud = HUD(args.width, args.height)
        self.client.load_world('Town01')
        self.wld = self.client.get_world()
        settings = self.wld.get_settings()
        settings.fixed_delta_seconds = Config.simulation_step
        settings.synchronous_mode = Config.synchronous
        self.wld.apply_settings(settings)

        self.scene_generator = Scenario(self.wld)

        wld_map = self.wld.get_map()
        print(wld_map.name)

        self.conn = Connector(self.despot_port)
        self.conn.establish_connection()
        m = self.conn.receive_message()
        print(m)  # START

    def get_benchmark(self):
        for scenario, speed, distance in self.exp_setting:
            scene = self.scene_generator.scenario10()
            world = World(self.wld, self.hud, scene, self.args)
            controller = KeyboardControl(world)
            agent = ISDespot(world, self.wld.get_map(), self.conn, scene)
            clock = pygame.time.Clock()

            while True:
                clock.tick_busy_loop(60)
                if controller.parse_events():
                    return
                world.tick(clock)
                if self.args.display:
                    world.render(self.display)
                    pygame.display.flip()

                control = agent.run_step()
                if control == "goal":
                    break
                world.player.apply_control(control)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.audi.tt")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--display',
        default=True,
        type=bool,
        help='Render the simulation window (default: False)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        despot = RunISDespot([1, 5], [10, 20], args, 2)
        despot.get_benchmark()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    subprocess.run(['cd /opt/carla-simulator/ && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)
    # subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True)


if __name__ == '__main__':
    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main()
