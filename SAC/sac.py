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
# from SAC.model import GaussianPolicy, QNetwork
from SAC.model_categorical import GaussianPolicy, QNetwork
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

    def select_action(self, state, evaluate=False):
        cat_tensor = state[1].type(torch.FloatTensor).to(self.device)
        state = torch.FloatTensor(state[0]).to(self.device)
        if evaluate is False:
            action, logp, _ = self.policy.sample((state, cat_tensor))
        else:
            _, logp, action = self.policy.sample((state, cat_tensor))
        action = action.detach().cpu().numpy()[0]
        return action

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, cat_batch, action_batch, reward_batch, next_state_batch, next_cat_batch, mask_batch = memory.sample(batch_size)
        state_batch = state_batch.to(self.device)
        cat_batch = cat_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        next_cat_batch = next_cat_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample((next_state_batch, next_cat_batch))
            next_state_log_pi = next_state_log_pi.unsqueeze(1)
            qf1_next_target, qf2_next_target = self.critic_target((next_state_batch, next_cat_batch),
                                                                  next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic((state_batch, cat_batch), action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample((state_batch, cat_batch))
        log_pi = log_pi.unsqueeze(1)
        for p in self.critic.parameters():
            p.requires_grad = False

        qf1_pi, qf2_pi = self.critic((state_batch, cat_batch), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True

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

    def update_parameters_categorical(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, cat_batch, action_batch, reward_batch, next_state_batch, next_cat_batch, mask_batch = memory.sample(batch_size)
        state_batch = state_batch.to(self.device)
        cat_batch = cat_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        next_cat_batch = next_cat_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)

        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.policy.sample((next_state_batch, next_cat_batch))

            qf1_next_target, qf2_next_target = self.critic_target((next_state_batch, next_cat_batch))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target = (next_state_action_probs * min_qf_next_target).sum(dim=1, keepdim=True)
            # print("Target: ", min_qf_next_target.size(), reward_batch.size())
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic((state_batch, cat_batch))
        qf1 = qf1.gather(1, action_batch.long())
        qf2 = qf2.gather(1, action_batch.long())
        # print("Network: ", qf1.size())
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        _, log_pi, pi = self.policy.sample((state_batch, cat_batch))
        log_pi = log_pi.unsqueeze(1)
        for p in self.critic.parameters():
            p.requires_grad = False

        qf1_pi, qf2_pi = self.critic((state_batch, cat_batch))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        q = torch.sum(min_qf_pi * pi, dim=1, keepdim=True)
        entropies = -torch.sum(pi * log_pi, dim=1, keepdim=True)
        policy_loss = -(q + self.alpha * entropies).mean()
        print("Entropy: {:.4f}, Q: {:.4f}".format(entropies.mean().item(), q.mean().item()))

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True

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
