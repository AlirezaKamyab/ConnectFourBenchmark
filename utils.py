import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from environment import Connect4Env
from models.cnn import CNN
from models.cnn_actor import ActorCriticCNN
from models.mlp import MLP
from models.mlp_actor import ActorMLP, ActorCriticMLP


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.0):
        super(SharedAdam, self).__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for param_group in self.param_groups:
            for p in param_group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0.0)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()

import torch
import torch.optim as optim

class SharedRMSprop(optim.RMSprop):
    def __init__(
        self, 
        params, 
        lr: float = 1e-4, 
        alpha: float = 0.99, 
        eps: float = 1e-5, 
        weight_decay: float = 0.0, 
        momentum: float = 0.0, 
        centered: bool = False
    ):
        super(SharedRMSprop, self).__init__(
            params, 
            lr=lr, 
            alpha=alpha, 
            eps=eps, 
            weight_decay=weight_decay, 
            momentum=momentum, 
            centered=centered
        )

        # Pre-allocate optimizer states and move them to shared memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['square_avg'] = torch.zeros_like(p.data)
                
                state['step'].share_memory_()
                state['square_avg'].share_memory_()

                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum_buffer'].share_memory_()

                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data)
                    state['grad_avg'].share_memory_()


def epsilon_greedy(
    values: torch.Tensor,
    epsilon: float,
    action_mask: np.ndarray = None
):
    # Ensure 2D tensor/array shape [batch_size, num_actions]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    
    if values.ndim == 1:
        values = values.reshape(1, -1)
        
    batch_size, num_actions = values.shape

    if action_mask is not None:
        action_mask = np.array(action_mask, dtype=np.float32).reshape(batch_size, num_actions)
        # Apply mask: set invalid actions to a large negative number
        masked_values = np.where(action_mask == 1, values, -1e9)
    else:
        action_mask = np.ones((batch_size, num_actions))
        masked_values = values

    actions = []
    for b in range(batch_size):
        valid_actions = np.where(action_mask[b] == 1)[0]
        if len(valid_actions) == 0:
            # Fallback if state is terminal/empty
            actions.append(0)
            continue

        if np.random.rand() < epsilon:
            # Random valid action
            action = np.random.choice(valid_actions)
        else:
            # Best valid action
            action = valid_actions[np.argmax(masked_values[b, valid_actions])]
        actions.append(action)

    return np.array(actions)


def convert_to_tensor(obs: dict, device: str = "cpu"):
    obs, action_mask = obs["observation"], obs["action_mask"]
    tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor, action_mask


def linear_epsilon_scheduler(maximum: float, minimum: float, num_steps: int):
    slope = (minimum - maximum) / num_steps

    def get_epsilon(step: int):
        new_epsilon = slope * step + maximum
        new_epsilon = max(minimum, new_epsilon)
        return new_epsilon

    return get_epsilon


def draw_plot(values: np.ndarray, title: str, x_label: str, y_label: str):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(values)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(title, fontdict={"fontsize": 16, "fontweight": 800})
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid("#eee")

    return ax, fig


def get_model(config: dict):
    model_type = config.get("type")
    assert (
        model_type is not None
    ), "Config file is corrupted. There is no type for the model"

    if model_type.lower() == "mlp":
        model = MLP(
            in_features=84,
            hidden_state=config["hidden_states"],
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
            num_actions=config.get("num_actions", None),
        )
        return model

    elif model_type.lower() == 'actor_mlp':
        model = ActorMLP(
            in_features=84,
            hidden_state=config["hidden_states"],
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
            num_actions=config.get("num_actions", None),
        )
        return model

    elif model_type.lower() == 'actor_critic_mlp':
        model = ActorCriticMLP(
            in_features=84,
            hidden_state=config["hidden_states"],
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
            num_actions=config.get("num_actions", None),
        )
        return model
    
    elif model_type.lower() == "cnn":
        model = CNN(
            in_channels=2,
            channels=config["channels"],
            kernels=config["kernels"],
            num_actions=config.get("num_actions", None),
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
        )
        return model

    elif model_type.lower() == "actor_cnn":
        model = CNN(
            in_channels=2,
            channels=config["channels"],
            kernels=config["kernels"],
            num_actions=config.get("num_actions", None),
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
        )
        return model

    elif model_type.lower() == "actor_critic_cnn":
        model = ActorCriticCNN(
            in_channels=2,
            channels=config["channels"],
            kernels=config["kernels"],
            num_actions=config.get("num_actions", None),
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
        )
        return model

    raise ValueError(f"The model={model_type} has not been defined")


def get_epsilon_scheduler(config: dict):
    if config is None:
        return None
    scheduler_type = config.get("type", None)
    if scheduler_type is None:
        return None

    if scheduler_type.lower() == "linear":
        func = linear_epsilon_scheduler(
            maximum=config.get("maximum", 1.0),
            minimum=config.get("minimum", 0.1),
            num_steps=config.get("steps", 500),
        )
        return func

    raise ValueError("This scheduler has not been defined")


def save_csv_file(column_names: list, array: np.ndarray, path: str):
    values = {column_names[i]: array[:, i] for i in range(len(column_names))}
    df = pd.DataFrame(values)
    df.to_csv(path, index=False)


@torch.no_grad
def play_a_game(
    env: Connect4Env, model: nn.Module, device: str = "cuda"
):
    model.eval()

    state = env.reset()
    state, action_mask = convert_to_tensor(state, device=device)
    frames = [env.render()]

    best_moves = 0
    total_moves = 0
    terminated = False
    while not terminated:
        values = model(state)
        action = epsilon_greedy(values=values, epsilon=0.0, action_mask=action_mask)[0]
        if action in env.get_all_best_actions():
            best_moves += 1
        total_moves += 1

        state, reward, terminated = env.step(action)['player_0']
        frames.append(env.render())
        state, action_mask = convert_to_tensor(state, device=device)

        # opponent's turn
        mini_max_action = env.predict_best_move()
        state, reward, terminated = env.step(mini_max_action)['player_0']
        frames.append(env.render())
        state, action_mask = convert_to_tensor(state, device=device)

    optimal_rate = best_moves / total_moves

    return {
        "mean_optimal_rate": optimal_rate,
        "outcome": reward,
        "frames":frames
    }


@torch.no_grad()
def final_evaluation(
    env: Connect4Env, model: nn.Module, runs: int = 1, device: str = "cuda"
):
    model.eval()
    optimal_rates = []
    wdl = []
    for _ in tqdm(range(runs)):
        state = env.reset()
        state, action_mask = convert_to_tensor(state, device=device)
        best_moves = 0
        total_moves = 0
        terminated = False
        while not terminated:
            values = model(state)
            action = epsilon_greedy(values=values, epsilon=0.0, action_mask=action_mask)
            if action in env.get_all_best_actions():
                best_moves += 1
            total_moves += 1
            state, reward, terminated, _, _ = env.step(action)
            state, action_mask = convert_to_tensor(state, device=device)

        if reward > 0:
            wdl.append(1)
        elif reward < 0:
            wdl.append(-1)
        else:
            wdl.append(0)
        optimal_rates.append(best_moves / total_moves)

    wdl = np.array(wdl, dtype=np.float32)
    optimal_rates = np.array(optimal_rates, dtype=np.float32)
    win_rate = np.sum(wdl == 1) / wdl.shape[0]
    draw_rate = np.sum(wdl == 0) / wdl.shape[0]
    lose_rate = np.sum(wdl == -1) / wdl.shape[0]

    return {
        "wdl": wdl,
        "mean_optimal_rate": optimal_rates.mean(),
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "lose_rate": lose_rate,
    }


@torch.no_grad()
def final_evaluation_actor(
    env: Connect4Env, model: nn.Module, runs: int = 1, device: str = "cpu"
):
    model.eval()
    optimal_rates = []
    wdl = []
    for _ in tqdm(range(runs)):
        state = env.reset()
        state, action_mask = convert_to_tensor(state, device=device)
        best_moves = 0
        total_moves = 0
        terminated = False
        while not terminated:
            logits, _ = model(state)

            # prepare mask
            action_mask = torch.tensor(action_mask, dtype=torch.bool)
            action_mask = ~action_mask

            # mask the logits
            masked_logits = logits.masked_fill(action_mask, -torch.inf)
            probs = torch.nn.functional.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().detach().item()

            if action in env.get_all_best_actions():
                best_moves += 1
            total_moves += 1
            state, reward, terminated = env.step(action)['player_0']
            state, action_mask = convert_to_tensor(state, device=device)

            state, next_reward, terminated = env.step(env.predict_best_move())['player_0']
            state, action_mask = convert_to_tensor(state, device=device)

            reward = reward + next_reward

        if reward > 0:
            wdl.append(1)
        elif reward < 0:
            wdl.append(-1)
        else:
            wdl.append(0)
        optimal_rates.append(best_moves / total_moves)

    wdl = np.array(wdl, dtype=np.float32)
    optimal_rates = np.array(optimal_rates, dtype=np.float32)
    win_rate = np.sum(wdl == 1) / wdl.shape[0]
    draw_rate = np.sum(wdl == 0) / wdl.shape[0]
    lose_rate = np.sum(wdl == -1) / wdl.shape[0]

    return {
        "wdl": wdl,
        "mean_optimal_rate": optimal_rates.mean(),
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "lose_rate": lose_rate,
    }


def compute_grad_norm(model:nn.Module, norm_type:float=2.0):
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue

        total += param.grad.norm(norm_type).mean()

    total = total ** (1 / norm_type)
    return total
