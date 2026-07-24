import json
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch

from environment import Connect4Env
from q_learning import q_learning
from utils import (
    draw_plot,
    final_evaluation,
    get_epsilon_scheduler,
    get_model,
    save_csv_file,
)

if __name__ == "__main__":
    parser = ArgumentParser(description="Q-learning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config_file = json.load(config_file)
        config = SimpleNamespace(**config_file)

    DEVICE = config.device
    experiment_name = config.experiment
    result_path = os.path.join("results", experiment_name)
    os.makedirs(result_path, exist_ok=True)

    env = Connect4Env(render_mode="rgb_array", simple_reward=True)
    env.load_openning_book("connect_four_solver/7x6.book")

    runs = config.runs
    episodes = config.episodes
    gamma = config.gamma
    alpha = config.alpha
    epsilon = config.epsilon
    epsilon_scheduler = get_epsilon_scheduler(config.epsilon_scheduler)
    history = None

    for run in range(runs):
        print("\nRun:", run + 1)
        model = get_model(config.q_network).to(DEVICE)
        output = q_learning(
            env=env,
            q_network=model,
            episodes=episodes,
            gamma=gamma,
            alpha=alpha,
            device=DEVICE,
            epsilon=epsilon,
            epsilon_scheduler=epsilon_scheduler,
        )

        if history is None:
            history = output
            continue

        history["td_errors"] += output["td_errors"]
        history["optimal_move_rates"] += output["optimal_move_rates"]
        history["win_rate"] += output["win_rate"]
        history["draw_rate"] += output["draw_rate"]
        history["lose_rate"] += output["lose_rate"]
        history["game_lengths"] += output["game_lengths"]

    for k in history.keys():
        history[k] /= runs

    _, td_error_fig = draw_plot(
        history["td_errors"], "TD-Error per Episode", "episode", "td-error"
    )
    td_error_fig.savefig(os.path.join(result_path, "td_errors.jpeg"), dpi=320)

    _, optimal_move_rate_fig = draw_plot(
        history["optimal_move_rates"],
        "Optimal move rate per Episode",
        "episode",
        "Optimal Move Rate",
    )
    optimal_move_rate_fig.savefig(
        os.path.join(result_path, "optimal_move_rate.jpeg"), dpi=320
    )

    _, game_lengths = draw_plot(
        history["game_lengths"], "No. actions per Episode", "episode", "No. actions"
    )
    game_lengths.savefig(os.path.join(result_path, "game_lengths.jpeg"), dpi=320)

    all_states = np.hstack(
        [
            history["td_errors"][:, None],
            history["optimal_move_rates"][:, None],
            history["game_lengths"][:, None],
        ]
    )
    save_csv_file(
        ["td_errors", "optimal_move_rates", "game_lengths"],
        all_states,
        os.path.join(result_path, "states.csv"),
    )

    with open(os.path.join(result_path, "win_rates.txt"), "w") as file:
        print("Configs", file=file)
        print("-" * 10, file=file)
        print(json.dumps(config_file), file=file)

        print("Train (Red Player) Results", file=file)
        print("-" * 10, file=file)
        print(f"Train Win-rate: {history['win_rate']:<2.2f}", file=file)
        print(f"Train Draw-rate: {history['draw_rate']:<2.2f}", file=file)
        print(f"Train Lose-rate: {history['lose_rate']:<2.2f}", file=file)
        print(file=file)

        eval_result = final_evaluation(env=env, model=model, runs=runs, device=DEVICE)

        print("Eval (Red Player) Results", file=file)
        print("-" * 10, file=file)
        print(f"Win-rate: {eval_result['win_rate']:<2.2f}", file=file)
        print(f"Draw-rate: {eval_result['draw_rate']:<2.2f}", file=file)
        print(f"Lose-rate: {eval_result['lose_rate']:<2.2f}", file=file)
        print(f"Mean optimal-rate: {eval_result['mean_optimal_rate']:<2.3f}", file=file)
        print(file=file)

    torch.save(model, os.path.join(result_path, "model.pt"))
