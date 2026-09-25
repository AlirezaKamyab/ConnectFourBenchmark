import json
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from environment import Connect4Env
from utils import (
    get_model,
    SharedAdam,
    SharedRMSprop
)
from actor_critic_async import worker as actor_critic

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
    actor_critic_net = get_model(config.actor)

    actor_critic_net.share_memory()
    history = None

    optimizer = SharedRMSprop(actor_critic_net.parameters(), lr=3e-4)
    num_workers = 12

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=actor_critic,
            args=(
                worker_id,
                config.actor,
                actor_critic_net,
                optimizer,
                1.0,
                100_000,
                10,
                1e-2
            )
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(actor_critic_net, 'checkpoint.pt')