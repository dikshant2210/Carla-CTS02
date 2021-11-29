"""
Author: Dikshant Gupta
Time: 28.11.21 23:42
"""

import os
import random

import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from agents.rl.sac.model import GaussianPolicy, QNetwork
from config import Config


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = Config.sac_gamma
        self.tau = Config.sac_tau
        self.alpha = Config.sac_alpha

        self.target_update_interval = Config.target_update_interval
        self.automatic_entropy_tuning = Config.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.n, Config.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=Config.sac_lr)

        self.critic_target = QNetwork(num_inputs, action_space.n, Config.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.n).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=Config.sac_lr)

        self.policy = GaussianPolicy(num_inputs, action_space.n, Config.hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=Config.sac_lr)

    def select_action(self, state, hidden, evaluate=False):
        cat_tensor = state[1].type(torch.FloatTensor).to(self.device)
        state = torch.FloatTensor(state[0]).to(self.device)
        lens = torch.IntTensor([1])
        if evaluate is False:
            action, _, _, hidden = self.policy.sample((state, cat_tensor), lens, hidden)
        else:
            _, _, action, hidden = self.policy.sample((state, cat_tensor), lens, hidden)
        action = action.detach().cpu().numpy()[0]
        return action.squeeze(0), hidden

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        episodes = random.sample(memory, batch_size)
        s = list()
        c = list()
        a = list()
        r = list()
        ns = list()
        nc = list()
        m = list()
        lens = list()
        for i in range(batch_size):
            state_batch, cat_batch, action_batch, reward_batch, next_state_batch, next_cat_batch, mask_batch = episodes[i].get_tensor_episode()
            s.append(state_batch)
            c.append(cat_batch)
            a.append(action_batch)
            r.append(reward_batch)
            ns.append(next_state_batch)
            nc.append(next_cat_batch)
            m.append(mask_batch)
            lens.append(state_batch.size()[0])

        state_batch = torch.nn.utils.rnn.pad_sequence(s, batch_first=True).to(self.device)
        cat_batch = torch.nn.utils.rnn.pad_sequence(c, batch_first=True).to(self.device)
        action_batch = torch.nn.utils.rnn.pad_sequence(a, batch_first=True).to(self.device)
        reward_batch = torch.nn.utils.rnn.pad_sequence(r, batch_first=True).to(self.device)
        next_state_batch = torch.nn.utils.rnn.pad_sequence(ns, batch_first=True).to(self.device)
        next_cat_batch = torch.nn.utils.rnn.pad_sequence(nc, batch_first=True).to(self.device)
        mask_batch = torch.nn.utils.rnn.pad_sequence(m, batch_first=True).to(self.device)
        # print(state_batch.size(), cat_batch.size(), action_batch.size(), reward_batch.size(), next_state_batch.size(), next_cat_batch.size(), mask_batch.size())
        lens = torch.tensor(lens)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample((next_state_batch, next_cat_batch), lens)
            qf1_next_target, qf2_next_target = self.critic_target((next_state_batch, next_cat_batch),
                                                                  next_state_action, lens)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic((state_batch, cat_batch), action_batch, lens)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _, _ = self.policy.sample((state_batch, cat_batch), lens)

        qf1_pi, qf2_pi = self.critic((state_batch, cat_batch), pi, lens)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
