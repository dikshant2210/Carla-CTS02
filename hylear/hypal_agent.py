"""
Author: Dikshant Gupta
Time: 14.12.21 22:23
"""


import os
import numpy as np
import torch
from torch.optim import Adam

from config import Config
from sac_discrete.base import BaseAgent
from sac_discrete.sacd.model import DQNBase, TwinnedQNetwork, CateoricalPolicy
from sac_discrete.sacd.utils import disable_gradients


class SharedSacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000, save_interval=100000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps, save_interval,
            log_interval, eval_interval, cuda, seed)

        # Define networks.
        self.conv = DQNBase(
            self.env.observation_space.shape[2]).to(self.device)
        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n,
            shared=True).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=dueling_net, shared=True).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=dueling_net, shared=True).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(
            list(self.conv.parameters()) +
            list(self.online_critic.Q1.parameters()), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state, probs=None):
        # Act with randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        probs = torch.FloatTensor(probs[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action, _, _ = self.policy.sample(state, probs)
            curr_q1 = self.online_critic.Q1(state)
            curr_q2 = self.online_critic.Q2(state)
            q = torch.min(curr_q1, curr_q2)
            critic_action = torch.argmax(q, dim=1)
        return action.item(), critic_action.item()

    def exploit(self, state):
        # Act without randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        nearmiss = False
        accident = False
        goal = False
        state = self.env.reset()
        action_count = {0: 0, 1: 0, 2: 0}
        action_count_critic = {0: 0, 1: 0, 2: 0}

        t = np.zeros(6)  # reward, vx, vt, onehot last action
        t[3 + 1] = 1.0  # index = 3 + last_action(maintain)

        while (not done) and episode_steps < self.max_episode_steps:
            if self.display:
                self.env.render()
            if self.start_steps > self.steps and False:
                action = self.env.action_space.sample()
                critic_action = action
            else:
                if self.env.control.throttle > 0:
                    symbolic_action = 0
                elif self.env.control.brake > 0:
                    symbolic_action = 2
                else:
                    symbolic_action = 1
                symbolic_probs = torch.FloatTensor([0.1, 0.1, 0.1])
                symbolic_probs[symbolic_action] = 0.8
                action, critic_action = self.explore((state, t), symbolic_probs)

            next_state, reward, done, info = self.env.step(action)
            action_count[action] += 1
            action_count_critic[critic_action] += 1

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)
            if episode_steps + 1 == self.max_episode_steps:
                mask = False
            else:
                mask = done
            # mask = False if episode_steps + 1 == self.max_episode_steps else done

            t_new = np.zeros(6)
            t_new[0] = clipped_reward
            t_new[1] = info['velocity'].x / Config.max_speed
            t_new[2] = info['velocity'].y / Config.max_speed
            t_new[3 + action] = 1.0

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append((state, t), action, clipped_reward, (next_state, t_new), mask)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state
            t = t_new
            nearmiss = nearmiss or info['near miss']
            accident = accident or info['accident']
            goal = info['goal']
            done = done or accident

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.steps % self.eval_interval == 0:
                self.evaluate()

            # if self.steps % self.save_interval == 0:
            #     self.save_models(os.path.join(self.model_dir, str(self.steps)))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.steps)

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            self.episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, accident, nearmiss))
        print('Total steps: {}, Episode steps: {}, Reward: {:.4f}'.format(self.steps, episode_steps, episode_return))
        print("Policy; ", action_count, "Critic: ", action_count_critic, "Alpha: {:.4f}".format(self.alpha.item()))

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        states, t = states
        states = self.conv(states)
        states = torch.cat([states, t], dim=-1)
        curr_q1 = self.online_critic.Q1(states).gather(1, actions.long())
        curr_q2 = self.online_critic.Q2(states.detach()).gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_states, t_new = next_states
            next_states = self.conv(next_states)
            next_states = torch.cat([next_states, t_new], dim=1)
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        states, t = states

        with torch.no_grad():
            states = self.conv(states)
        states = torch.cat([states, t], dim=1)

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.conv.save(os.path.join(save_dir, 'conv.pth'))
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
