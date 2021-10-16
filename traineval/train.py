"""
Author: Dikshant Gupta
Time: 16.10.21 09:31
"""

import torch
# import torch.functional as F
import torch.nn.functional as F
from agents.rl.a2c.model import A2C

import carla
import pygame
import argparse
import logging
import subprocess
import time
import pickle as pkl
import random
import numpy as np
from multiprocessing import Process
from world import World
from hud import HUD
from agents.navigation.rlagent import RLAgent
from agents.navigation.config import Config
from agents.tools.scenario import Scenario
from traineval.traineval_utils import KeyboardControl


def test_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    client = None

    # try:
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    if args.display:
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        # display.fill((0, 0, 0))
        pygame.display.flip()

    hud = HUD(args.width, args.height)
    client.load_world('Town01')
    wld = client.get_world()
    settings = wld.get_settings()
    settings.fixed_delta_seconds = Config.simulation_step
    settings.synchronous_mode = Config.synchronous
    wld.apply_settings(settings)

    scene_generator = Scenario(wld)
    scene = scene_generator.scenario10()
    world = World(wld, hud, scene, args)
    controller = KeyboardControl(world)
    # world.camera_manager.toggle_recording()

    wld_map = wld.get_map()
    print(wld_map.name)

    clock = pygame.time.Clock()
    while True:
        clock.tick_busy_loop(60)

        if controller.parse_events():
            return

        # agent.update_information()
        world.tick(clock)
        if args.display:
            world.render(display)
            pygame.display.flip()

        control = carla.VehicleControl()
        control.steer = 0
        control.throttle = 0.5
        if Config.synchronous:
            frame_num = wld.tick()
        if control == "goal":
            break
        world.player.apply_control(control)

    # finally:

    if world and world.recording_enabled:
        client.stop_recorder()

    if world is not None:
        world.destroy()

    pygame.quit()


def train_a2c(args):
    pygame.init()
    pygame.font.init()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    if args.display:
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

    hud = HUD(args.width, args.height)
    client.load_world('Town01')
    wld = client.get_world()
    settings = wld.get_settings()
    settings.fixed_delta_seconds = Config.simulation_step
    settings.synchronous_mode = Config.synchronous
    wld.apply_settings(settings)

    scene_generator = Scenario(wld)
    scene = scene_generator.scenario10()
    world = World(wld, hud, scene, args)
    controller = KeyboardControl(world)

    wld_map = wld.get_map()
    print(wld_map.name)

    current_episode = 0
    max_episodes = 2
    while current_episode < max_episodes:
        print("Episode: {}".format(current_episode + 1))
        world.restart(scene)
        clock = pygame.time.Clock()
        for _ in range(args.num_steps):
            clock.tick_busy_loop(60)

            if controller.parse_events():
                return

            world.tick(clock)
            if args.display:
                world.render(display)
                pygame.display.flip()

            control = carla.VehicleControl()
            control.steer = 0
            control.throttle = 0.5
            if Config.synchronous:
                frame_num = wld.tick()
            if control == "goal":
                break
            world.player.apply_control(control)

        current_episode += 1

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
        '--display',
        default=True,
        type=bool,
        help='Render the simulation window (default: False)')
    argparser.add_argument(
        '--lr', type=float,
        default=0.0001,
        help='learning rate (default: 0.0001)')
    argparser.add_argument(
        '--gamma', type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    argparser.add_argument(
        '--gae-lambda', type=float,
        default=1.00,
        help='lambda parameter for GAE (default: 1.00)')
    argparser.add_argument(
        '--entropy-coef', type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    argparser.add_argument(
        '--value-loss-coef', type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    argparser.add_argument(
        '--max-grad-norm', type=float,
        default=50,
        help='value loss coefficient (default: 50)')
    argparser.add_argument(
        '--seed', type=int,
        default=1,
        help='random seed (default: 1)')
    argparser.add_argument(
        '--num-processes', type=int,
        default=4,
        help='how many training processes to use (default: 4)')
    argparser.add_argument(
        '--num-steps', type=int,
        default=300,
        help='number of forward steps in A3C (default: 20)')
    argparser.add_argument(
        '--max-episode-length', type=int,
        default=1000000,
        help='maximum length of an episode (default: 1000000)')
    argparser.add_argument(
        '--env-name',
        default='PongDeterministic-v4',
        help='environment to train on (default: PongDeterministic-v4)')
    argparser.add_argument(
        '--no-shared',
        default=False,
        help='use an optimizer without shared momentum.')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        train_a2c(args)
        # test_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    subprocess.run(['cd /opt/carla-simulator/ && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)
    # subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True)


if __name__ == '__main__':
    # p = Process(target=run_server)
    # p.start()
    # time.sleep(5)  # wait for the server to start

    main()
    # p.terminate()
