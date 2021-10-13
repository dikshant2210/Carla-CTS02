"""
Author: Dikshant Gupta
Time: 06.10.21 04:51
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


def train_a2c(args):
    pygame.init()
    pygame.font.init()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    display = None
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
    scene = scene_generator.scenario01()
    world = World(wld, hud, scene, args)
    controller = KeyboardControl(world)

    wld_map = wld.get_map()
    print(wld_map.name)
    agent = RLAgent(world, wld.get_map(), scene)

    episodes = list()
    for scenario in Config.scenarios:
        for speed in np.arange(Config.ped_speed_range[0], Config.ped_speed_range[1] + 1, 0.1):
            for distance in np.arange(Config.ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                episodes.append((scenario, speed, distance))

    random.shuffle(episodes)

    torch.manual_seed(100)
    model = A2C(hidden_dim=256, num_actions=3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    episode_length, current_episode = 0, 0
    done = False
    max_episodes = min(10000, len(episodes))
    print("Total number of episodes: {}".format(max_episodes))
    while current_episode < max_episodes:
        # TODO: Select scenario and instance, reset the world
        scenario_id, ped_speed, ped_distance = episodes[current_episode]
        func = 'scene_generator.scenario' + scenario_id
        scenario = eval(func+'()')
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, "
              "Ped_distance: {:.2f}m".format(current_episode+1, scenario_id, ped_speed, ped_distance))
        world.restart(scenario, ped_speed, ped_distance)

        cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        if not done:
            hx = hx.detach()
            cx = cx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        clock = pygame.time.Clock()
        clock.tick_busy_loop(60)

        if controller.parse_events():
            return

        world.tick(clock)
        if args.display:
            world.render(display)
            pygame.display.flip()

        # Simulating the scenario instance
        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
        observation = None
        for step in range(args.num_steps):
            # TODO: get observation
            control, observation = agent.run_step()
            # with open("temp_check/obs.pkl", "wb") as file:
            #     pkl.dump(observation, file)
            # print(observation.shape)
            episode_length += 1
            input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x, velocity_y, speed_action])).cuda().type(
                torch.cuda.FloatTensor)
            logit, value, (hx, cx) = model(input_tensor, (hx, cx), cat_tensor)

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            # TODO: Replace final action with output from network
            speed_action = action.cpu().numpy()[0][0]
            if speed_action == 0:
                control.throttle = 0.6
            elif speed_action == 2:
                control.throttle = -0.6
            else:
                control.throttle = 0
            if Config.synchronous:
                frame_num = wld.tick()
            if control == "goal":
                done = True
            world.player.apply_control(control)
            # TODO: get the next observation
            _, observation = agent.run_step()
            velocity = agent.vehicle.get_velocity()
            velocity_x = velocity.x
            velocity_y = velocity.y
            reward, goal = agent.get_reward()
            # print(step, speed_action, reward)
            # observation, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length or goal
            # reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                # observation = env.reset()

            observation = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1).cuda().type(torch.cuda.FloatTensor)
        if not done:
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x, velocity_y, speed_action])).cuda().type(
                torch.cuda.FloatTensor)
            _, value, _ = model(observation, (hx, cx), cat_tensor)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).cuda().type(torch.cuda.FloatTensor)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        current_episode = current_episode + 1

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
        default=40,
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
