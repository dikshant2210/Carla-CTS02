"""
Author: Dikshant Gupta
Time: 22.06.21 11:52
"""

from agents.navigation.agent import Agent
from carla import VehicleControl


class HylearAgent(Agent):

    def run_step(self, debug=False):
        control = VehicleControl()
        control.throttle = 0.9
        return control


class GidasExperiment:
    def __init__(self):
        pass
