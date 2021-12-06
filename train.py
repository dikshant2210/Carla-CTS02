import argparse
import numpy as np
import torch
import subprocess
import pygame
import time
from multiprocessing import Process

from SAC.sac import SAC
from SAC.replay_memory import Memory
from environment import GIDASBenchmark
from config import Config

batch_size = Config.batch_size * 300


class SACTrainer:
    def __init__(self, args):
        seed = 123456
        self.env = GIDASBenchmark()
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Agent
        self.agent = SAC(self.env.observation_space.shape[0], self.env.action_space, args)

        # Memory
        self.episode_memory = Memory()

    def train(self):
        # Training Loop
        total_numsteps = 0
        updates = 0
        max_episodes = Config.train_episodes
        print("Total training episode: {}".format(max_episodes))
        for i_episode in range(max_episodes):
            episode_reward = 0
            episode_steps = 0
            state = self.env.reset()
            state = torch.tensor(state).float().unsqueeze(0)

            a = 1
            reward = 0
            velocity_x = 0
            velocity_y = 0
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6, a])).view(1, 4)
            nearmiss = False
            acccident = False

            for _ in range(Config.num_steps):
                if Config.pre_train_steps > total_numsteps:
                    # action = np.zeros(self.env.action_space.n)
                    action = self.env.action_space.sample()
                    # action[a] = 1.0  # Sample random action
                else:
                    # Sample action from policy
                    with torch.no_grad():
                        action = self.agent.select_action((state, cat_tensor))
                    # a = np.argmax(action, axis=-1)

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
                self.episode_memory.add(state, cat_tensor, action, reward, next_state, next_cat_tensor, mask)
                state = next_state
                cat_tensor = next_cat_tensor
                nearmiss = nearmiss or info['near miss']
                acccident = acccident or info['accident']

                if done or info['accident']:
                    break

            if total_numsteps > Config.total_training_steps:
                break

            print("Episode: {}, Scenario: {}, Ped. Speed: {:.2f}, Ped Distance: {}".format(
                i_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
            print("Total numsteps: {}, episode steps: {}, reward: {}".format(
                total_numsteps, episode_steps, round(episode_reward, 4)))
            print("Goal: {}, Accident: {}, Nearmiss: {}".format(info['goal'], acccident, nearmiss))

            if len(self.episode_memory.state) > batch_size and total_numsteps > Config.pre_train_steps:
                # Number of updates per step in environment
                for i in range(Config.update_freq):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters_categorical(
                        self.episode_memory, batch_size, updates)
                    updates += 1
                    print("Q-Loss: {:.4f}, Policy Loss: {:.4f}".format(critic_2_loss + critic_1_loss, policy_loss))

        self.env.close()


def main(args):
    print(__doc__)

    try:
        sac_trainer = SACTrainer(args)
        sac_trainer.train()

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
        help='TCP port to listen to (default: 2800)')
    arg_parser.add_argument(
        '-ckp', '--checkpoint',
        default='',
        type=str,
        help='load the model from this checkpoint')
    arg_parser.add_argument('--cuda', action="store_true",
                            help='run on CUDA (default: False)')
    arg = arg_parser.parse_args()
    Config.port = arg.port

    p = Process(target=run_server)
    p.start()
    time.sleep(5)  # wait for the server to start

    main(arg)
