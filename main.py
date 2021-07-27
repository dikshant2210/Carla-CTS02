"""
Author: Dikshant Gupta
Time: 23.03.21 14:55
"""

import carla
import pygame
import argparse
import logging
import time
import subprocess
import random
from multiprocessing import Process
from world import World
# from control import KeyboardControl
from hud import HUD
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.hylear_agent import HyLEAR
from agents.tools.connector import Connector

from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q, K_r
from pygame.locals import K_TAB


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.world = world
        for i in range(3):
            self.world.camera_manager.toggle_camera()

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    self.world.camera_manager.toggle_camera()
                elif event.key == K_r:
                    self.world.camera_manager.toggle_recording()

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    client = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        client.load_world('Town01')
        wld = client.get_world()
        world = World(wld, hud, args)
        print(wld.get_map())
        # controller = KeyboardControl(world, args.autopilot)
        controller = KeyboardControl(world)

        agent = BehaviorAgent(world.player, behavior='normal')
        spawn_points = world.map.get_spawn_points()
        random.shuffle(spawn_points)
        if spawn_points[0].location != agent.vehicle.get_location():
            destination = spawn_points[0].location
        else:
            destination = spawn_points[1].location
        agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            # if controller.parse_events(client, world, clock):
            #     return
            if controller.parse_events():
                return

            agent.update_information()
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if len(agent.get_local_planner().waypoints_queue) == 0:
                print("Target reached, mission accomplished...")

            speed_limit = world.player.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step()
            world.player.apply_control(control)

    finally:

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def game_loop_hylear(args):
    pygame.init()
    pygame.font.init()
    world = None
    client = None
    despot_port = 1245

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        client.load_world('Town01')
        wld = client.get_world()
        world = World(wld, hud, args)
        controller = KeyboardControl(world)

        # agent = BehaviorAgent(world.player, behavior='normal')
        # spawn_points = world.map.get_spawn_points()
        # random.shuffle(spawn_points)
        # if spawn_points[0].location != agent.vehicle.get_location():
        #     destination = spawn_points[0].location
        # else:
        #     destination = spawn_points[1].location
        # agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        wld_map = wld.get_map()
        # odr_world = client.generate_opendrive_world(wld_map.to_opendrive())

        conn = Connector(despot_port)
        agent = HyLEAR(world, wld.get_map(), conn)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            # if controller.parse_events(client, world, clock):
            #     return
            if controller.parse_events():
                return

            # agent.update_information()
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # if len(agent.get_local_planner().waypoints_queue) == 0:
            #     print("Target reached, mission accomplished...")

            # speed_limit = world.player.get_speed_limit()
            # agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step()
            if control == "goal":
                break
            world.player.apply_control(control)

    finally:

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


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
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        # game_loop(args)
        game_loop_hylear(args)

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
