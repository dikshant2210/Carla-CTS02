import torch
import random
from collections import deque
from config import Config


class Memory:
    def __init__(self):
        Config.episode_buffer = 20000
        self.state = deque([], maxlen=Config.episode_buffer)
        self.cat_tensor = deque([], maxlen=Config.episode_buffer)
        self.action = deque([], maxlen=Config.episode_buffer)
        self.reward = deque([], maxlen=Config.episode_buffer)
        self.next_state = deque([], maxlen=Config.episode_buffer)
        self.next_cat_tensor = deque([], maxlen=Config.episode_buffer)
        self.mask = deque([], maxlen=Config.episode_buffer)
        self.weight = []

    def add(self, state, cat_tensor, action, reward, next_state, next_cat_tensor, mask):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.mask.append(mask)
        self.cat_tensor.append(cat_tensor)
        self.next_cat_tensor.append(next_cat_tensor)
        self.weight.append(1.0)

    def sample(self, batch_size):
        total = sum(self.weight)
        weights = [x / total for x in self.weight]
        indexes = random.choices(list(range(len(self.state))), weights=weights, k=batch_size)
        state = []
        cat_tensor = []
        action = []
        reward = []
        next_state = []
        next_cat_tensor = []
        mask = []
        for i in indexes:
            state.append(self.state[i])
            cat_tensor.append(self.cat_tensor[i])
            action.append(self.action[i])
            reward.append(self.reward[i])
            next_state.append(self.next_state[i])
            next_cat_tensor.append(self.next_cat_tensor[i])
            mask.append(self.mask[i])

        state = torch.vstack([s for s in state])
        action = torch.vstack([torch.from_numpy(a) for a in action]).type(torch.FloatTensor)
        reward = torch.vstack([torch.tensor(r) for r in reward]).type(torch.FloatTensor)
        next_state = torch.vstack([ns for ns in next_state])
        mask = torch.vstack([torch.tensor(m) for m in mask])
        cat_tensor = torch.vstack([c for c in cat_tensor]).type(torch.FloatTensor)
        next_cat_tensor = torch.vstack([nc for nc in next_cat_tensor]).type(torch.FloatTensor)
        return state, cat_tensor, action, reward, next_state, next_cat_tensor, mask
