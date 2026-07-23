from environment import Connect4Env
import torch
from torch import nn
from torch.optim import SGD
import numpy as np
from typing import Callable
from utils import (
    convert_to_tensor, 
    epsilon_greedy, 
    draw_plot, 
    final_evaluation,
    get_model,
    save_csv_file,
    linear_epsilon_scheduler
)
import os
from types import SimpleNamespace
from argparse import ArgumentParser
import json


def q_learning(
    env:Connect4Env,
    q_network:nn.Module,
    episodes:int,
    gamma:float,
    alpha:float=0.1,
    epsilon:float=0.1,
    device:str='cpu',
    epsilon_scheduler:Callable=None
):
    sgd = SGD(q_network.parameters(), lr=alpha)
    sgd.zero_grad()
    global_step = 0
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
                    next_q_values:torch.Tensor = q_network(next_obs)
                    next_action = epsilon_greedy(next_q_values, epsilon=0, action_mask=next_action_mask)
                    max_q = next_q_values[:, next_action]
                Ut = reward + gamma * max_q
            else:
                Ut = torch.tensor([reward], dtype=torch.float32, device=device)
                outcomes.append(reward)

            loss = nn.functional.mse_loss(q_values[:, action], Ut)

            with torch.no_grad():
                td_error:torch.Tensor = Ut - q_values[:, action]
                sum_td_errors += td_error.detach().cpu().abs().item()

            # apply updates
            loss.backward()
            sgd.step()
            sgd.zero_grad()

            obs = next_obs
            action_mask = next_action_mask
            global_step += 1
            if epsilon_scheduler is not None:
                epsilon = epsilon_scheduler(global_step)

        optimal_move_rate.append(best_moves / total_moves)
        game_lengths.append(total_moves)
        td_errors.append(sum_td_errors)

        # Log
        w = np.sum(np.array(outcomes) > 0)
        d = np.sum(np.array(outcomes) == 0)
        l = np.sum(np.array(outcomes) < 0)
        print(f"\rEpisode {episode + 1:<4}, W/D/L: {w:<4}|{d:<4}|{l:<4} Td-error: {float(td_errors[-1]):2.3f}", end='')

    outcomes = np.array(outcomes, np.float32)
    win_rate = np.sum(outcomes == 1) / outcomes.shape[0]
    draw_rate = np.sum(outcomes == 0) / outcomes.shape[0]
    lose_rate = np.sum(outcomes == -1) / outcomes.shape[0]
    
    return {
        "win_rate":win_rate,
        "draw_rate":draw_rate,
        "lose_rate":lose_rate,
        "td_errors":np.array(td_errors),
        "optimal_move_rates":np.array(optimal_move_rate),
        "game_lengths":np.array(game_lengths, np.float32)
    }


if __name__ == '__main__':
    parser = ArgumentParser(description="Q-learning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config_file = json.load(config_file)
        config = SimpleNamespace(**config_file)

    DEVICE = config.device
    experiment_name = config.experiment
    result_path = os.path.join('results', experiment_name)
    os.makedirs(result_path, exist_ok=True)

    env = Connect4Env(render_mode="rgb_array", simple_reward=True)
    env.load_openning_book("connect_four_solver/7x6.book")

    runs = config.runs
    episodes = config.episodes
    gamma = config.gamma
    alpha = config.alpha
    epsilon = config.epsilon
    history = None

    for run in range(runs):
        print('\nRun:', run+1)
        mlp = get_model(config.q_network).to(DEVICE)
        output = q_learning(
            env=env,
            q_network=mlp,
            episodes=episodes, 
            gamma=gamma,
            alpha=alpha,
            device=DEVICE,
            epsilon=epsilon,
            epsilon_scheduler=linear_epsilon_scheduler(1.0, 0.1, 1000)
        )

        if history is None:
            history = output
            continue

        history['td_errors'] += output['td_errors']
        history['optimal_move_rates'] += output['optimal_move_rates']
        history['win_rate'] += output['win_rate']
        history['draw_rate'] += output['draw_rate']
        history['lose_rate'] += output['lose_rate']
        history['game_lengths'] += output['game_lengths']

    for k in history.keys():
        history[k] /= runs

    _, td_error_fig = draw_plot(history['td_errors'], "TD-Error per Episode", "episode", "td-error")
    td_error_fig.savefig(os.path.join(result_path, 'td_errors.jpeg'), dpi=320)

    _, optimal_move_rate_fig = draw_plot(history['optimal_move_rates'], "Optimal move rate per Episode", "episode", "Optimal Move Rate")
    optimal_move_rate_fig.savefig(os.path.join(result_path, 'optimal_move_rate.jpeg'), dpi=320)

    _, game_lengths = draw_plot(history['game_lengths'], "No. actions per Episode", "episode", "No. actions")
    game_lengths.savefig(os.path.join(result_path, 'game_lengths.jpeg'), dpi=320)

    all_states = np.hstack([history['td_errors'][None, :], history['optimal_move_rates'][None, :], history['game_lengths'][None, :]])
    save_csv_file(
        ["td_errors", 'optimal_move_rates', 'game_lengths'], 
        all_states,
        os.path.join(result_path, 'states.csv')
    )

    with open(os.path.join(result_path, "win_rates.txt"), 'w') as file:
        print("Configs", file=file)
        print("-"*10, file=file)
        print(json.dumps(config_file), file=file)


        print("Train (Red Player) Results", file=file)
        print("-"*10, file=file)
        print(f"Train Win-rate: {history['win_rate']:<2.2f}", file=file)
        print(f"Train Draw-rate: {history['draw_rate']:<2.2f}", file=file)
        print(f"Train Lose-rate: {history['lose_rate']:<2.2f}", file=file)
        print(file=file)

        eval_result = final_evaluation(
            env=env,
            model=mlp,
            runs=runs,
            device=DEVICE
        )

        print("Eval (Red Player) Results", file=file)
        print("-"*10, file=file)
        print(f"Win-rate: {eval_result['win_rate']:<2.2f}", file=file)
        print(f"Draw-rate: {eval_result['draw_rate']:<2.2f}", file=file)
        print(f"Lose-rate: {eval_result['lose_rate']:<2.2f}", file=file)
        print(f"Mean optimal-rate: {eval_result['mean_optimal_rate']:<2.3f}", file=file)
        print(file=file)
