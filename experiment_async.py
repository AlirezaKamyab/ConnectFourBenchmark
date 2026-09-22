import json
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from environment import Connect4Env
from utils import (
    epsilon_greedy,
    get_epsilon_scheduler,
    get_model,
    play_a_game,
    convert_to_tensor,
    SharedAdam
)
from async_q_learning import worker as q_learning_async

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = ArgumentParser(description="Q-learning")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config_file = json.load(config_file)
        config = SimpleNamespace(**config_file)

    DEVICE = config.device
    experiment_name = config.experiment
    result_path = os.path.join("results", experiment_name)
    os.makedirs(result_path, exist_ok=True)


    game_env = Connect4Env(render_mode='rgb_array')
    game_env.load_openning_book("connect_four_solver/7x6.book")

    gamma = config.gamma
    alpha = config.alpha
    epsilon = config.epsilon
    play_each_n_steps = config.play_each_n_steps
    steps_to_swap_target = config.steps_to_swap_target
    model = get_model(config.q_network)
    target = get_model(config.q_network)
    target.load_state_dict(model.state_dict())
    target.eval().requires_grad_(False)
    model.share_memory()
    target.share_memory()

    steps = torch.tensor(0, dtype=torch.int32)
    steps.share_memory_()
    steps_to_swap = torch.tensor(0, dtype=torch.int32)
    steps_to_swap.share_memory_()
    history = None

    optimizer = SharedAdam(model.parameters(), lr=1e-4)
    num_workers = 8

    processes = []
    epsilons = [0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.05, 0.2]
    for worker_id in range(num_workers):
        p = mp.Process(
            target=q_learning_async,
            args=(
                worker_id,
                steps,
                steps_to_swap,
                config.q_network,
                model,
                target,
                optimizer,
                epsilons[worker_id],
                10000,
                0.99,
                5
            )
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(model, 'checkpoint.pt')