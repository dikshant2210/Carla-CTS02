"""
Author: Dikshant Gupta
Time: 10.12.21 04:14
"""

import subprocess
import argparse
import time
from multiprocessing import Process

from environment import GIDASBenchmark
from rainbow.rainbow import Rainbow
from config import Config


def train(args):
    env = GIDASBenchmark()
    rainbow_trainer = Rainbow(env)
    rainbow_trainer.train(batch_size=args.batch_size)


def run_server():
    port = "-carla-port={}".format(Config.port)
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl ' + port], shell=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    arg_parser.add_argument(
        '-b', '--batch_size',
        metavar='B',
        default=16,
        type=int,
        help='Batch Size (default: 16)')
    arg = arg_parser.parse_args()

    p = Process(target=run_server)
    p.start()
    time.sleep(5)

    train(arg)
