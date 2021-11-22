"""
Author: Dikshant Gupta
Time: 24.10.21 00:47
"""

import subprocess
from multiprocessing import Process
import numpy as np
import os
import random
import pygame
import carla
import argparse
import time
import logging
import torch
from datetime import datetime
from torch.optim import Adam
import torch.nn.functional as F

from config import Config
from environment.environment import GIDASBenchmark
from agents.rl.sac.model import SAC, QNetwork
from agents.rl.sac.utils import soft_update, hard_update, ExperienceBuffer


def train_sac():
    ##############################################################
    t0 = time.time()
    # Logging file
    filename = "_out/sac/{}.log".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    print(filename)
    file = open(filename, "w")
    file.write(str(vars(Config)) + "\n")

    # Path to save model
    path = "_out/sac/"
    if not os.path.exists(path):
        os.mkdir(path)

    # Setting up environment
    env = GIDASBenchmark()

    # Instantiating RL agent
    torch.manual_seed(100)
    rl_agent = SAC(Config.num_actions).cuda()
    critic_optim = Adam(list(rl_agent.q_network.parameters()) + list(rl_agent.shared_network.parameters()), lr=args.lr)
    critic_target = QNetwork(Config.num_actions).cuda()
    hard_update(critic_target, rl_agent.q_network)
    policy_optim = Adam(list(rl_agent.action_policy.parameters()) + list(rl_agent.shared_network.parameters()),
                        lr=args.lr)

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
    current_episode = 0
    max_episodes = Config.train_episodes
    print("Total training episodes: {}".format(max_episodes))
    file.write("Total training episodes: {}\n".format(max_episodes))

    while current_episode < max_episodes:
        m = env.reset()

        # Setup placeholders for training value logs
        d = False
        rAll = 0
        j = 0
        episode_buffer = []
        total_episode_reward = 0

        # Setup initial inputs for LSTM Cell
        cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
        nearmiss = False
        accident = False
        goal = False

        for step_num in range(Config.num_steps):
            if total_steps == Config.pre_train_steps:
                print("Pre-train phase done.")

            j += 1

            # Forward pass of the RL Agent
            m = torch.from_numpy(m).cuda().type(torch.cuda.FloatTensor)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x, velocity_y, speed_action])).cuda().type(
                torch.cuda.FloatTensor)
            m = torch.reshape(m, (-1, 1, 100, 100))
            cat_tensor = torch.reshape(cat_tensor, (-1, 4))
            with torch.no_grad():
                state1 = rl_agent.shared_network(m, hx, cx, cat_tensor)
                a, _, _ = rl_agent.action_policy.sample(state1[0])

            a = a.cpu().numpy()
            speed_action = np.argmax(a, axis=-1)[0]
            m1, r, done, info = env.step(speed_action)

            m1 = torch.from_numpy(m1).cuda().type(torch.cuda.FloatTensor)
            m1 = torch.reshape(m1, (-1, 1, 100, 100))
            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            goal = info['goal']
            nearmiss = nearmiss or info['nearmiss']
            accident = accident or info['accident']
            reward = r
            d = goal or accident

            total_episode_reward += r * (Config.y ** j)
            total_steps += 1

            episode_buffer.append(np.reshape(np.array([m.cpu(), a, r, m1.cpu(), hx.cpu(), cx.cpu(),
                                                       state1[0].cpu(), state1[1].cpu(), not d,
                                                       cat_tensor.cpu()]), [1, 10]))

            if total_steps > Config.pre_train_steps and current_episode + 1 > Config.batch_size + 4:
                if e > Config.endE:
                    e -= step_drop

                if total_steps % Config.update_freq == 0:

                    # Get a random batch of experiences.
                    train_batch = my_buffer.sample(Config.batch_size, Config.trace_length)

                    # Run a forward pass of rl_agent and update parameters
                    state_batch = torch.from_numpy(np.vstack(train_batch[:, 0] / 1.0)).cuda().type(
                        torch.cuda.FloatTensor)
                    action_batch = torch.from_numpy(np.vstack(train_batch[:, 1] / 1.0)).cuda().type(
                        torch.cuda.FloatTensor)
                    reward_batch = torch.from_numpy(np.vstack(train_batch[:, 2] / 1.0)).cuda().type(
                        torch.cuda.FloatTensor)
                    next_state_batch = torch.from_numpy(np.vstack(train_batch[:, 3] / 1.0)).cuda().type(
                        torch.cuda.FloatTensor)
                    lstm_state_batch_hx = torch.from_numpy(np.vstack(train_batch[:, 4])).cuda().type(
                        torch.cuda.FloatTensor)
                    lstm_state_batch_cx = torch.from_numpy(np.vstack(train_batch[:, 5])).cuda().type(
                        torch.cuda.FloatTensor)
                    next_lstm_state_batch_hx = torch.from_numpy(np.vstack(train_batch[:, 6])).cuda().type(
                        torch.cuda.FloatTensor)
                    next_lstm_state_batch_cx = torch.from_numpy(np.vstack(train_batch[:, 7])).cuda().type(
                        torch.cuda.FloatTensor)
                    mask_batch = torch.from_numpy(np.vstack(train_batch[:, 8] / 1.0)).cuda().type(
                        torch.cuda.FloatTensor)
                    cat_batch = torch.from_numpy(np.vstack(train_batch[:, 9] / 1.0)).cuda().type(
                        torch.cuda.FloatTensor)

                    with torch.no_grad():
                        features, _ = rl_agent.shared_network(next_state_batch, next_lstm_state_batch_hx,
                                                              next_lstm_state_batch_cx, cat_batch)
                        next_state_action, next_state_log_pi, _ = rl_agent.action_policy.sample(features)
                        qf1_next_target, qf2_next_target = critic_target(features, next_state_action)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        next_q_value = reward_batch + mask_batch * gamma * min_qf_next_target

                    # Two Q-functions to mitigate positive bias in the policy improvement step
                    f, _ = rl_agent.shared_network(state_batch, lstm_state_batch_hx, lstm_state_batch_cx, cat_batch)
                    qf1, qf2 = rl_agent.q_network(f, action_batch)

                    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                    qf1_loss = F.mse_loss(qf1, next_q_value)
                    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                    qf2_loss = F.mse_loss(qf2, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    pi, log_pi, _ = rl_agent.action_policy.sample(f)
                    qf1_pi, qf2_pi = rl_agent.q_network(f, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    critic_optim.zero_grad()
                    policy_optim.zero_grad()
                    loss = qf_loss + policy_loss
                    loss.backward()
                    critic_optim.step()
                    policy_optim.step()

                    soft_update(critic_target, rl_agent.q_network, tau)
                    print("Q-loss: {:.4f}, Policy loss: {:.4f}".format(qf_loss.detach().cpu(),
                                                                       policy_loss.detach().cpu()))
                    # print("Updating Network Weight!")

            rAll += r
            hx = state1[0]
            cx = state1[1]
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

        print("Goal reached: {}, Near miss: {}, Crash: {}".format(goal, nearmiss, accident))
        if current_episode + 1 % Config.save_freq == 0:
            torch.save(rl_agent.state_dict(), "{}sac_{}.pth".format(Config.path, current_episode))

    if current_episode + 1 % Config.save_freq == 0:
        torch.save(rl_agent.state_dict(), "{}sac_{}.pth".format(Config.path, current_episode))


def main(args):
    print(__doc__)

    try:
        train_sac()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


def run_server():
    port = "-carla-port={}".format(Config.port)
    subprocess.run(['cd /home/carla && SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl ' + port], shell=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    arg_parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    arg_parser.add_argument(
        '-ckp', '--checkpoint',
        default='',
        type=str,
        help='load the model from this checkpoint')
    arg = arg_parser.parse_args()
    Config.port = arg.port

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main(arg)
    p.terminate()
