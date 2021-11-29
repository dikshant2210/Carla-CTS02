import torch


class EpisodeMemory:
    def __init__(self):
        self.state = []
        self.cat_tensor = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.next_cat_tensor = []
        self.mask = []

    def add(self, state, cat_tensor, action, reward, next_state, next_cat_tensor, mask):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.mask.append(mask)
        self.cat_tensor.append(cat_tensor)
        self.next_cat_tensor.append(next_cat_tensor)

    def get_tensor_episode(self):
        state = torch.vstack([s for s in self.state])
        action = torch.vstack([torch.from_numpy(a) for a in self.action]).type(torch.FloatTensor)
        reward = torch.vstack([torch.tensor(r) for r in self.reward]).type(torch.FloatTensor)
        next_state = torch.vstack([ns for ns in self.next_state])
        mask = torch.vstack([torch.tensor(m) for m in self.mask])
        cat_tensor = torch.vstack([c for c in self.cat_tensor]).type(torch.FloatTensor)
        next_cat_tensor = torch.vstack([nc for nc in self.next_cat_tensor]).type(torch.FloatTensor)
        return state, cat_tensor, action, reward, next_state, next_cat_tensor, mask
