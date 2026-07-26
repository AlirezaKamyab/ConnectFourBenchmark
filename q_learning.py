import torch
from torch import nn
from torch.optim import Adam

from utils import epsilon_greedy, compute_grad_norm
from torch.utils.tensorboard import SummaryWriter


class Q_Learning:
    def __init__(
        self,
        q_network: nn.Module,
        gamma: float,
        alpha: float = 0.1,
        device: str = "cpu",
        log_dir:str=None
    ):
        self.q_network = q_network
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        self.sgd = Adam(q_network.parameters(), lr=alpha)
        self.global_steps = 0

        if log_dir is not None:
            self.logger = SummaryWriter(log_dir=log_dir)


    def step(
        self,
        *,
        state:torch.Tensor,
        action:torch.Tensor,
        next_state:torch.Tensor,
        next_action_mask:torch.Tensor,
        reward:torch.Tensor,
        terminated:torch.Tensor
    ):
        q_values = self.q_network(state)
        q_values = torch.gather(q_values, 1, index=action)
        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            next_actions = epsilon_greedy(
                next_q_values, 
                epsilon=0, 
                action_mask=next_action_mask.cpu().numpy(),
            )
            next_actions = torch.tensor(next_actions, dtype=torch.long, device=self.device)
            next_q_values = torch.gather(next_q_values, 1, next_actions.unsqueeze(1))
            Ut = reward + (1 - terminated) * self.gamma * next_q_values
            td_error = Ut - q_values

            if self.logger:
                self.logger.add_scalar("train/td_error", td_error.cpu().abs().mean().item(), self.global_steps)

        loss = nn.functional.mse_loss(q_values, Ut)
        loss.backward()
        self.sgd.step()

        if self.logger:
            grad_norm = compute_grad_norm(self.q_network)
            self.logger.add_scalar("train/grad_norm", grad_norm.cpu().item(), self.global_steps)

        self.sgd.zero_grad()
        self.global_steps += 1


if __name__ == '__main__':
    pass