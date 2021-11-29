import argparse
import numpy as np
import itertools
import torch
import gym
from collections import deque
import multiprocessing

from agents.rl.sac.sac import SAC
from agents.rl.sac.replay_memory import EpisodeMemory
from environment import GIDASBenchmark
from config import Config


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


class Counter(object):
    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class SACTrainer:
    def __init__(self):
        seed = 123456
        self.env = GIDASBenchmark()
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Agent
        self.agent = SAC(self.env.observation_space.shape[0], self.env.action_space, args)

        # Memory
        manager = multiprocessing.Manager()
        shared_list = manager.list()
        self.storage = deque(shared_list, maxlen=1)
        self.total_steps = Counter(0)

    def train(self):
        # Training Loop
        total_numsteps = 0
        updates = 0
        running_reward = 0.05
        max_episodes = Config.train_episodes
        for i_episode in range(max_episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            hidden = None
            state = self.env.reset()
            state = torch.tensor(state).float().unsqueeze(0)
            episode_memory = EpisodeMemory()

            a = 1
            reward = 0
            velocity_x = 0
            velocity_y = 0
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6, a])).view(1, 4)
            nearmiss = False
            acccident = False

            for _ in range(Config.num_steps):
                if Config.pre_train_steps > total_numsteps:
                    action = np.zeros(self.env.action_space.n)
                    a = self.env.action_space.sample()
                    action[a] = 1.0  # Sample random action
                else:
                    # Sample action from policy
                    action, hidden = self.agent.select_action((state.unsqueeze(0), cat_tensor.unsqueeze(0)), hidden)
                    a = np.argmax(action, axis=-1)

                next_state, reward, done, info = self.env.step(a)  # Step
                next_state = torch.tensor(next_state).float().unsqueeze(0)
                velocity = info['velocity']
                velocity_x = velocity.x
                velocity_y = velocity.y
                next_cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6, a])).view(1, 4)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                mask = 1 if episode_steps == Config.num_steps else float(not done)

                # Append transition to memory
                episode_memory.add(state, cat_tensor, action, reward, next_state, next_cat_tensor, mask)

                state = next_state
                cat_tensor = next_cat_tensor

            self.storage.append(episode_memory)
            if total_numsteps > Config.total_training_steps:
                break

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            print("Episode: {}, Scenario: {}, Ped. Speed: {}, Ped Distance: {}".format(
                i_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
            print("Total numsteps: {}, episode steps: {}, reward: {}, avg. reward: {:.3f}".format(
                total_numsteps, episode_steps, round(episode_reward, 2), running_reward))

            if len(self.storage) > Config.batch_size:
                # Number of updates per step in environment
                print("Updating parametrs!")
                for i in range(Config.update_freq):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(
                        self.storage, Config.batch_size, updates)
                    updates += 1

            # if i_episode % 50 == 0:
            #     self.eval()

        self.env.close()

    def eval(self):
        avg_reward = 0.
        episodes = 5
        for _ in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            hidden = None
            while not done:
                action, hidden = self.agent.select_action(torch.tensor(state).float().view(1, 1, -1), hidden,
                                                          evaluate=True)
                a = np.argmax(action, axis=-1)
                next_state, reward, done, _ = self.env.step(a)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")


def train_sac():
    sac_trainer = SACTrainer()
    sac_trainer.train()


if __name__ == '__main__':
    train_sac()
