import random
import numpy as np


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ExperienceBuffer:
    def __init__(self, buffer_size=300):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        sampled_traces = np.reshape(sampled_traces, [batch_size * trace_length, 10])
        indices = [i for i in range(len(sampled_traces))]
        random.shuffle(indices)
        return sampled_traces[indices]
