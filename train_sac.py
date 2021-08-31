"""
Author: Dikshant Gupta
Time: 30.08.21 18:42
"""

import os
import numpy as np

from agents.rl.config import Config
from agents.rl.connector import Connector
from agents.rl.model import SAC, QNetwork
from agents.rl.utils import soft_update, hard_update, ExperienceBuffer
import csv
import argparse
import torch
from datetime import datetime
from torch.optim import Adam
import torch.functional as F


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
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


total = ""

PORT = 4001

myBuffer = ExperienceBuffer()

# Set the rate of random action decrease.
e = Config.startE
stepDrop = (Config.startE - Config.endE) / Config.anneling_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(Config.path):
    os.makedirs(Config.path)

# Write the first line of the master log-file for the Control Center
with open(Config.path + '/log.csv', 'w+') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

connection = Connector()
model = SAC(Config.num_actions)

gamma = args.gamma
tau = args.tau
alpha = args.alpha

device = torch.device("cuda" if args.cuda else "cpu")
critic_optim = Adam(list(model.q_network.parameters()) + list(model.shared_network.parameters()), lr=args.lr)
critic_target = QNetwork(Config.num_actions).to(device)
hard_update(critic_target, model.q_network)
policy_optim = Adam(list(model.action_policy.parameters()) + list(model.shared_network.parameters()), lr=args.lr)

if Config.load_model:
    # TODO: load pretrained model
    pass

for i in range(Config.num_episodes):
    episodeBuffer = []
    # Reset environment and get first new observation

    totalEpisodeReward = 0
    connection.sendMessage("RESET\n")

    message = connection.receiveMessage()
    s = connection.one_hot(Config.num_actions) + [message['angle']]
    s.extend(message['obs'])
    s = np.array(s)
    m = np.array(message['map'])
    connection.sendMessage("START\n")

    d = False
    rAll = 0
    j = 0
    state = (np.zeros([1, Config.hidden_size]),
             np.zeros([1, Config.hidden_size]))  # Reset the recurrent layer's hidden state

    while j < Config.max_epLength:
        if total_steps == Config.pre_train_steps:
            print("Pre-train phase done.")

        j += 1

        with torch.no_grad():
            state1 = model.shared_network((m, torch.FloatTensor(torch.from_numpy(state)).to(device)))
            a, _, _ = model.action_policy.sample(state1[0])

        # TODO: argmax of a
        a = a.detach().numpy()
        connection.sendMessage(str(a) + "\n")

        message = connection.receiveMessage()
        s1 = connection.one_hot(a) + [message['angle']]
        s1.extend(message['obs'])
        s1 = np.array(s1)
        m1 = np.array(message['map'])
        r = message['reward']
        d = message['terminal']

        totalEpisodeReward += r * (Config.y ** j)
        total_steps += 1

        # episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d, m, m1]), [1, 7]))
        episodeBuffer.append(np.reshape(np.array([m, a, r, m1, state, state1, not d]), [1, 7]))

        if total_steps > Config.pre_train_steps and i > Config.batch_size + 4:
            if e > Config.endE:
                e -= stepDrop

            if total_steps % Config.update_freq == 0:
                # Reset the recurrent layer's hidden state
                state_train = (np.zeros([Config.batch_size, Config.hidden_size]),
                               np.zeros([Config.batch_size, Config.hidden_size]))

                start_time = datetime.now()
                if total_steps % (Config.update_freq * 5) == 0:
                    start_time = datetime.now()

                # Get a random batch of experiences.
                trainBatch = myBuffer.sample(Config.batch_size, Config.trace_length)

                if total_steps % (Config.update_freq * 100) == 0:
                    time_elapsed = datetime.now() - start_time
                    print('Time elapsed for sampling from buffer (hh:mm:ss.ms) {}'.format(time_elapsed))
                    start_time = datetime.now()

                # TODO: Run a forward pass of model and update parameters
                state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 0] / 1.0))).to(device)
                action_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 1] / 1.0))).to(device)
                reward_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 2] / 1.0))).to(device)
                next_state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 3] / 1.0))).to(device)
                lstm_state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 4] / 1.0))).to(device)
                next_lstm_state_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 5] / 1.0))).to(device)
                mask_batch = torch.FloatTensor(torch.from_numpy(np.vstack(trainBatch[:, 6] / 1.0))).to(device)

                with torch.no_grad():
                    features, _ = model.shared_network((next_state_batch, next_lstm_state_batch))
                    next_state_action, next_state_log_pi, _ = model.action_policy.sample(features)
                    qf1_next_target, qf2_next_target = critic_target(features, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = reward_batch + mask_batch * gamma * min_qf_next_target

                # Two Q-functions to mitigate positive bias in the policy improvement step
                qf1, qf2 = model.q_network(state_batch, action_batch)

                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf1_loss = F.mse_loss(qf1,
                                      next_q_value)
                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2,
                                      next_q_value)
                qf_loss = qf1_loss + qf2_loss

                critic_optim.zero_grad()
                qf_loss.backward()
                critic_optim.step()

                f, _ = model.shared_network((state_batch, lstm_state_batch))
                pi, log_pi, _ = model.action_policy.sample(f)

                qf1_pi, qf2_pi = model.q_network(f, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                policy_optim.zero_grad()
                policy_loss.backward()
                policy_optim.step()

                soft_update(critic_target, model.q_network, tau)

                if total_steps % (Config.update_freq * 100) == 0:
                    time_elapsed = datetime.now() - start_time
                    print('Time elapsed for updating model (hh:mm:ss.ms) {}'.format(time_elapsed))

        rAll += r
        s = s1
        state = state1
        m = m1

        if d:
            break

    # Add the episode to the experience buffer
    if j > Config.trace_length:
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)
