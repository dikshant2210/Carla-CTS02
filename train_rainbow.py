"""
Author: Dikshant Gupta
Time: 10.12.21 04:14
"""

import subprocess
import argparse
import time
from multiprocessing import Process

from environment import GIDASBenchmark
from Rainbow.rainbow import Rainbow
from config import Config


def train():
    env = GIDASBenchmark()
    rainbow_trainer = Rainbow(env)
    rainbow_trainer.train(batch_size=4)


def run_server():
    port = "-carla-port={}".format(Config.port)
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl ' + port], shell=True)


if __name__ == '__main__':
    p = Process(target=run_server)
    p.start()
    time.sleep(5)

    train()
