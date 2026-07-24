from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import SGD

from environment import Connect4Env
from utils import convert_to_tensor, epsilon_greedy


def q_learning(
    env: Connect4Env,
    q_network: nn.Module,
    episodes: int,
    gamma: float,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    device: str = "cpu",
    epsilon_scheduler: Callable = None,
):
    sgd = SGD(q_network.parameters(), lr=alpha)
    sgd.zero_grad()
    outcomes = []
    td_errors = []
    optimal_move_rate = []
    game_lengths = []

    for episode in range(episodes):
        obs = env.reset()
        obs, action_mask = convert_to_tensor(obs, device=device)
        terminated = False

        best_moves = 0
        total_moves = 0
        sum_td_errors = 0
        while not terminated:
            q_values = q_network(obs)
            action = epsilon_greedy(q_values, epsilon, action_mask)

            # To compute the rate at which the agent chooses the optimal action
            if action in env.get_all_best_actions():
                best_moves += 1
            total_moves += 1

            next_obs, reward, terminated, _, _ = env.step(action)
            if env.get_player() == env.computer_player and terminated:
                reward *= -1
            next_obs, next_action_mask = convert_to_tensor(next_obs, device=device)

            # Ut should be [B,]
            if not terminated:
                with torch.no_grad():
                    next_q_values: torch.Tensor = q_network(next_obs)
                    next_action = epsilon_greedy(
                        next_q_values, epsilon=0, action_mask=next_action_mask
                    )
                    max_q = next_q_values[:, next_action]
                Ut = reward + gamma * max_q
            else:
                Ut = torch.tensor([reward], dtype=torch.float32, device=device)
                outcomes.append(reward)

            loss = nn.functional.mse_loss(q_values[:, action], Ut)

            with torch.no_grad():
                td_error: torch.Tensor = Ut - q_values[:, action]
                sum_td_errors += td_error.detach().cpu().abs().item()

            # apply updates
            loss.backward()
            sgd.step()
            sgd.zero_grad()

            obs = next_obs
            action_mask = next_action_mask

        if epsilon_scheduler is not None:
            epsilon = epsilon_scheduler(episode)

        optimal_move_rate.append(best_moves / total_moves)
        game_lengths.append(total_moves)
        td_errors.append(sum_td_errors)

        # Log
        w = np.sum(np.array(outcomes) > 0)
        d = np.sum(np.array(outcomes) == 0)
        l = np.sum(np.array(outcomes) < 0)
        print(
            f"\rEpisode {episode + 1:<4}, W/D/L: {w:<4}|{d:<4}|{l:<4} Td-error: {float(td_errors[-1]):2.3f}",
            end="",
        )

    outcomes = np.array(outcomes, np.float32)
    win_rate = np.sum(outcomes == 1) / outcomes.shape[0]
    draw_rate = np.sum(outcomes == 0) / outcomes.shape[0]
    lose_rate = np.sum(outcomes == -1) / outcomes.shape[0]

    return {
        "outcomes": outcomes,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "lose_rate": lose_rate,
        "td_errors": np.array(td_errors),
        "optimal_move_rates": np.array(optimal_move_rate),
        "game_lengths": np.array(game_lengths, np.float32),
    }
