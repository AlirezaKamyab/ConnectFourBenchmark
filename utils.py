import torch
import numpy as np
from torch import nn
from environment import Connect4Env
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.mlp import MLP
import pandas as pd


def epsilon_greedy(values:torch.Tensor, epsilon:float, action_mask:np.ndarray=None, batch:bool=False):
    values = values.detach().cpu().numpy()

    if action_mask is not None:
        actions = np.argwhere(action_mask == 1)[:, 0]
    else:
        actions = int(values.shape[-1])

    values = values * action_mask + np.ones_like(values) * -1000 * (1 - action_mask)

    batch_size = values.shape[0]
    greedy = np.argmax(values, axis=1)
    random = np.random.choice(actions, size=(batch_size,))
    mask = np.random.choice([0, 1], p=(epsilon, 1 - epsilon), size=(batch_size,))
    actions = greedy * mask + random * (1 - mask)

    if batch: return actions
    else: return actions[0]


def convert_to_tensor(obs:dict, device:str='cpu'):
    obs, action_mask = obs['observation'], obs['action_mask']
    tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor, action_mask


def linear_epsilon_scheduler(
    maximum:float,
    minimum:float,
    num_steps:int
):
    slope = (minimum - maximum) / num_steps
    def get_epsilon(step:int):
        new_epsilon = slope * step + maximum
        new_epsilon = max(minimum, new_epsilon)
        return new_epsilon
    return get_epsilon


def draw_plot(
    values:np.ndarray,
    title:str,
    x_label:str,
    y_label:str
):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(values)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title, fontdict={'fontsize':16, 'fontweight':800})
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid("#eee")

    return ax, fig


def get_model(config:dict):
    model_type = config.get("type")
    assert model_type is not None, "Config file is corrupted. There is no type for the model"

    if model_type.lower() == 'mlp':
        model = MLP(
            in_features=84,
            hidden_state=config['hidden_states'],
            bias=config.get("bias", True),
            activation_function=config.get("activation", "relu"),
            num_actions=7
        )
        return model

    raise ValueError("This model has not been defined")


def save_csv_file(column_names:list, array:np.ndarray, path:str):
    values = {
        column_names[i]:array[:, i]
        for i in range(len(column_names))
    }
    df = pd.DataFrame(values)
    df.to_csv(path, index=False)


@torch.no_grad()
def final_evaluation(
    env: Connect4Env,
    model:nn.Module,
    runs:int=1,
    device:str='cuda'
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
            action = epsilon_greedy(
                values=values,
                epsilon=0.0,
                action_mask=action_mask
            )
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
        "wdl":wdl,
        "mean_optimal_rate":optimal_rates.mean(),
        "win_rate":win_rate,
        "draw_rate":draw_rate,
        "lose_rate":lose_rate
    }
