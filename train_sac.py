"""
Author: Dikshant Gupta
Time: 24.10.21 00:47
"""

import subprocess
from multiprocessing import Process
import numpy as np
import random
import pygame
import carla
import argparse
import time
import torch
from datetime import datetime
from torch.optim import Adam
import torch.nn.functional as F

from config import Config
from agents.rl.sac.model import SAC, QNetwork
from agents.rl.sac.utils import soft_update, hard_update, ExperienceBuffer
from agents.navigation.rlagent import RLAgent
from world import World
from hud import HUD
from agents.tools.scenario import Scenario


def train_sac(args):
    ##############################################################
    # Setting up simulator and world configuration #
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
    scene = scene_generator.scenario01()
    world = World(wld, hud, scene, args)
    # controller = KeyboardControl(world)

    wld_map = wld.get_map()
    print(wld_map.name)
    ##############################################################

    ##############################################################
    # Compiling all different combination of scenarios
    episodes = list()
    for scenario in Config.scenarios:
        for speed in np.arange(Config.ped_speed_range[0], Config.ped_speed_range[1] + 1, 0.1):
            for distance in np.arange(Config.ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                episodes.append((scenario, speed, distance))
    random.shuffle(episodes)
    ##############################################################

    ##############################################################
    # Instantiating all the symbolic and RL agent
    planner_agent = RLAgent(world, wld.get_map(), scene)
    torch.manual_seed(100)
    rl_agent = SAC(Config.num_actions).cuda()
    device = torch.device("cuda" if args.cuda else "cpu")
    critic_optim = Adam(list(rl_agent.q_network.parameters()) + list(rl_agent.shared_network.parameters()), lr=args.lr)
    critic_target = QNetwork(Config.num_actions).to(device)
    hard_update(critic_target, rl_agent.q_network)
    policy_optim = Adam(list(rl_agent.action_policy.parameters()) + list(rl_agent.shared_network.parameters()), lr=args.lr)
    ##############################################################

    ##############################################################
    # Setting up placeholders for SAC training
    my_buffer = ExperienceBuffer()

    # Set the rate of random action decrease.
    e = Config.startE
    step_drop = (Config.startE - Config.endE) / Config.anneling_steps

    # create lists to contain total rewards and steps per episode
    j_list = []
    r_list = []
    total_steps = 0

    gamma = args.gamma
    tau = args.tau
    alpha = args.alpha
    ##############################################################

    ##############################################################
    # Simulation loop
    for current_episode in range(Config.num_episodes):
        ##############################################################
        # Get the scenario id, parameters and instantiate the world
        scenario_id, ped_speed, ped_distance = episodes[current_episode]
        func = 'scene_generator.scenario' + scenario_id
        scenario = eval(func + '()')
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, "
              "Ped_distance: {:.2f}m".format(current_episode + 1, scenario_id, ped_speed, ped_distance))
        world.restart(scenario, ped_speed, ped_distance)

        # Reset environment and get first new observation
        control, m = planner_agent.run_step()
        m = torch.from_numpy(m).cuda().type(torch.cuda.FloatTensor)

        d = False
        rAll = 0
        j = 0
        episode_buffer = []
        total_episode_reward = 0
        state = (np.zeros([1, Config.hidden_size]),
                 np.zeros([1, Config.hidden_size]))  # Reset the recurrent layer's hidden state
        ##############################################################

        ##############################################################
        clock = pygame.time.Clock()
        while j < Config.max_epLength:
            clock.tick_busy_loop(60)
            if total_steps == Config.pre_train_steps:
                print("Pre-train phase done.")

            j += 1

            ##############################################################
            # Forward pass of the RL Agent
            with torch.no_grad():
                state1 = rl_agent.shared_network((m, torch.FloatTensor(torch.from_numpy(state)).to(device)))
                a, _, _ = rl_agent.action_policy.sample(state1[0])

            a = a.detach().numpy()
            speed_action = np.argmax(a, axis=-1)
            if speed_action == 0:
                control.throttle = 0.6
            elif speed_action == 2:
                control.throttle = -0.6
            else:
                control.throttle = 0
            world.tick(clock)
            if args.display:
                world.render(display)
                pygame.display.flip()

            if Config.synchronous:
                frame_num = wld.tick()
            world.player.apply_control(control)

            _, m1 = planner_agent.run_step()
            m1 = torch.from_numpy(m1).cuda().type(torch.cuda.FloatTensor)
            velocity = planner_agent.vehicle.get_velocity()
            velocity_x = velocity.x
            velocity_y = velocity.y
            r, goal, accident = planner_agent.get_reward()
            d = goal or accident

            total_episode_reward += r * (Config.y ** j)
            total_steps += 1

            episode_buffer.append(np.reshape(np.array([m, a, r, m1, state, state1, not d]), [1, 7]))

            if total_steps > Config.pre_train_steps and current_episode > Config.batch_size + 4:
                if e > Config.endE:
                    e -= step_drop

                if total_steps % Config.update_freq == 0:
                    # Reset the recurrent layer's hidden state
                    state_train = (np.zeros([Config.batch_size, Config.hidden_size]),
                                   np.zeros([Config.batch_size, Config.hidden_size]))

                    start_time = datetime.now()
                    if total_steps % (Config.update_freq * 5) == 0:
                        start_time = datetime.now()

                    # Get a random batch of experiences.
                    trainBatch = my_buffer.sample(Config.batch_size, Config.trace_length)

                    if total_steps % (Config.update_freq * 100) == 0:
                        time_elapsed = datetime.now() - start_time
                        print('Time elapsed for sampling from buffer (hh:mm:ss.ms) {}'.format(time_elapsed))
                        start_time = datetime.now()

                    # TODO: Run a forward pass of rl_agent and update parameters
                    state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 0] / 1.0))).to(device)
                    action_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 1] / 1.0))).to(device)
                    reward_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 2] / 1.0))).to(device)
                    next_state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 3] / 1.0))).to(device)
                    lstm_state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 4] / 1.0))).to(device)
                    next_lstm_state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 5] / 1.0))).to(device)
                    mask_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 6] / 1.0))).to(device)

                    with torch.no_grad():
                        features, _ = rl_agent.shared_network((next_state_batch, next_lstm_state_batch))
                        next_state_action, next_state_log_pi, _ = rl_agent.action_policy.sample(features)
                        qf1_next_target, qf2_next_target = critic_target(features, next_state_action)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        next_q_value = reward_batch + mask_batch * gamma * min_qf_next_target

                    # Two Q-functions to mitigate positive bias in the policy improvement step
                    qf1, qf2 = rl_agent.q_network(state_batch, action_batch)

                    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                    qf1_loss = F.mse_loss(qf1, next_q_value)
                    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                    qf2_loss = F.mse_loss(qf2, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    critic_optim.zero_grad()
                    qf_loss.backward()
                    critic_optim.step()

                    f, _ = rl_agent.shared_network((state_batch, lstm_state_batch))
                    pi, log_pi, _ = rl_agent.action_policy.sample(f)

                    qf1_pi, qf2_pi = rl_agent.q_network(f, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    policy_optim.zero_grad()
                    policy_loss.backward()
                    policy_optim.step()

                    soft_update(critic_target, rl_agent.q_network, tau)

                    if total_steps % (Config.update_freq * 100) == 0:
                        time_elapsed = datetime.now() - start_time
                        print('Time elapsed for updating rl_agent (hh:mm:ss.ms) {}'.format(time_elapsed))

            rAll += r
            state = state1
            m = m1

            if d:
                break

        # Add the episode to the experience buffer
        if j > Config.trace_length:
            bufferArray = np.array(episode_buffer)
            episode_buffer = list(zip(bufferArray))
            my_buffer.add(episode_buffer)
            j_list.append(j)
            r_list.append(rAll)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='rl_agent updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    args = parser.parse_args()

    train_sac(args)


def run_server():
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl'], shell=True)
    # subprocess.run(['cd ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/ && ./car'], shell=True)


if __name__ == '__main__':
    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main()
    # p.terminate()
