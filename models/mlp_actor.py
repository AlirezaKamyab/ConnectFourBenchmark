import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorMLP(nn.Module):
    def __init__(
            self,
            in_features: int = 84,
            hidden_state: list = [512, 256],
            num_actions: int = None,
            bias: bool = False,
            activation_function: str = "relu",
        ):
            super(ActorMLP, self).__init__()
    
            num_actions = num_actions if num_actions is not None else 1
    
            if activation_function == "gelu":
                act = nn.GELU()
            elif activation_function == "silu":
                act = nn.SiLU()
            elif activation_function == "selu":
                act = nn.SELU()
            else:
                act = nn.ReLU()
    
            hidden_state = [in_features] + hidden_state + [num_actions]
            self.layers = nn.ModuleList()
            for i in range(1, len(hidden_state) - 1):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_state[i - 1], hidden_state[i], bias=bias),
                        act,
                    )
                )
    
            self.layers.append(nn.Linear(hidden_state[-2], hidden_state[-1]))

            for layer in self.layers:
                if hasattr(layer, 'weight'):
                    layer.weight.data.mul_(0.01)
                if hasattr(layer, 'bias'):
                    layer.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
        return x


class ActorCriticMLP(nn.Module):
    def __init__(
            self,
            in_features: int = 84,
            hidden_state: list = [512, 256],
            num_actions: int = None,
            bias: bool = False,
            activation_function: str = "relu",
        ):
            super(ActorCriticMLP, self).__init__()
    
            num_actions = num_actions if num_actions is not None else 1
    
            if activation_function == "gelu":
                act = nn.GELU()
            elif activation_function == "silu":
                act = nn.SiLU()
            elif activation_function == "selu":
                act = nn.SELU()
            else:
                act = nn.ReLU()
    
            hidden_state = [in_features] + hidden_state + [num_actions]
            self.layers = nn.ModuleList()
            for i in range(1, len(hidden_state) - 1):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_state[i - 1], hidden_state[i], bias=bias),
                        act,
                    )
                )
    
            self.actor_linear = nn.Linear(hidden_state[-2], hidden_state[-1])
            self.critic_linear = nn.Linear(hidden_state[-2], 1)

            for layer in self.layers:
                if hasattr(layer, 'weight'):
                    layer.weight.data.mul_(0.01)
                if hasattr(layer, 'bias'):
                    layer.bias.data.fill_(0.0)

            self.actor_linear.weight.data.mul_(0.01)
            self.actor_linear.bias.data.fill_(0.0)
            self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        actor_output = self.actor_linear(x)
        critic_output = self.critic_linear(x)
        return actor_output, critic_output