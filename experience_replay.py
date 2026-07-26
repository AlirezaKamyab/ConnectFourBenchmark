import torch
import numpy as np
from environment import Connect4Env
from utils import epsilon_greedy
from models.cnn import CNN
from q_learning import Q_Learning


class ExperienceReplay:
    def __init__(
            self, 
            max_size:int=50000,
            device:str='cpu'
        ):
        self.experience = []
        self.max_size = max_size
        self.device = device

    def add_experience(
        self,
        *,
        state:np.ndarray,
        action:int,
        next_state:np.ndarray,
        next_action_mask:np.ndarray,
        reward:float,
        terminated:bool
    ):
        if isinstance(state, torch.Tensor):
            state = state.detach().clone()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).reshape(1, 1)

        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().clone()
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(1, 1)
        terminated = torch.tensor(1 if terminated else 0, dtype=torch.float32).reshape(1, 1)
        self.experience.append(
            {
                'state':state, 
                'action':action, 
                'next_state':next_state, 
                'next_action_mask':next_action_mask,
                'reward':reward, 
                'terminated':terminated
            }
        )
        self.experience = self.experience[-self.max_size:]

    def __len__(self):
        return len(self.experience)

    def sample(
        self,
        batch_size:int
    ):
        max_idx = len(self.experience)
        indices = np.random.choice(max_idx, size=(batch_size,))

        state = [self.experience[i]['state'] for i in indices]
        action = [self.experience[i]['action'] for i in indices]
        next_state = [self.experience[i]['next_state'] for i in indices]
        next_action_mask = [self.experience[i]['next_action_mask'] for i in indices]
        reward = [self.experience[i]['reward'] for i in indices]
        terminated = [self.experience[i]['terminated'] for i in indices]

        state = torch.concat(state, dim=0).to(device=self.device)
        action = torch.concat(action, dim=0).to(device=self.device)
        next_state = torch.concat(next_state, dim=0).to(device=self.device)
        next_action_mask = torch.concat(next_action_mask, dim=0).to(device=self.device)
        reward = torch.concat(reward, dim=0).to(device=self.device)
        terminated = torch.concat(terminated, dim=0).to(device=self.device)

        return {
            "state":state,
            "action":action,
            "next_state":next_state,
            "next_action_mask":next_action_mask,
            "reward":reward,
            "terminated":terminated
        }