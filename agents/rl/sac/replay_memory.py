import torch


class EpisodeMemory:
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.mask = []

    def add(self, state, action, reward, next_state, mask):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.mask.append(mask)

    def get_tensor_episode(self):
        state = torch.vstack([s for s in self.state])
        action = torch.vstack([torch.from_numpy(a) for a in self.action]).type(torch.FloatTensor)
        reward = torch.vstack([torch.tensor(r) for r in self.reward]).type(torch.FloatTensor)
        next_state = torch.vstack([ns for ns in self.next_state])
        mask = torch.vstack([torch.tensor(m) for m in self.mask])
        return state, action, reward, next_state, mask
