import random
import torch
from collections import deque


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ExpBuffer:
    def __init__(self, max_storage, sample_length):
        self.sample_length = sample_length
        self.storage = deque([], maxlen=max_storage)

    def write_tuple(self, transition):
        self.storage.append(transition)

    def sample(self, batch_size):
        # Returns sizes of (batch_size * seq_len, *) depending on action/observation/return/done
        seq_len = self.sample_length * batch_size
        transitions = random.sample(self.storage, seq_len)

        obs = torch.vstack([tr[0] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        hx = torch.vstack([tr[1] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        cx = torch.vstack([tr[2] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        action = torch.vstack([torch.from_numpy(tr[3]) for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        rewards = torch.vstack([torch.tensor(tr[4]) for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        next_obs = torch.vstack([tr[5] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        next_hx = torch.vstack([tr[6] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        next_cx = torch.vstack([tr[7] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        cat = torch.vstack([tr[8] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        next_cat = torch.vstack([tr[9] for tr in transitions]).cuda().type(torch.cuda.FloatTensor)
        return obs, hx, cx, action, rewards, next_obs, next_hx, next_cx, cat, next_cat
