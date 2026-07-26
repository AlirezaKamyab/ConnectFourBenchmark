import json
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch

from environment import Connect4Env
from q_learning import Q_Learning
from experience_replay import ExperienceReplay
from utils import (
    draw_plot,
    final_evaluation,
    epsilon_greedy,
    get_epsilon_scheduler,
    get_model,
    play_a_game,
    convert_to_tensor
)

if __name__ == "__main__":
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

    env = Connect4Env(simple_reward=True)
    env.load_openning_book("connect_four_solver/7x6.book")

    game_env = Connect4Env(render_mode='rgb_array', simple_reward=True)
    game_env.load_openning_book("connect_four_solver/7x6.book")

    runs = config.runs
    gamma = config.gamma
    alpha = config.alpha
    epsilon = config.epsilon
    epsilon_scheduler = get_epsilon_scheduler(config.epsilon_scheduler)
    num_steps = config.num_steps
    batch_size = config.batch_size
    min_buffer_size = config.min_buffer_size
    max_buffer_size = config.max_buffer_size
    log_dir = config.log_dir
    play_each_n_steps = config.play_each_n_steps
    history = None

    for run in range(runs):
        buffer = ExperienceReplay(max_buffer_size, device=DEVICE)
        model = get_model(config.q_network).to(DEVICE)
        q_learning = Q_Learning(
            q_network=model,
            gamma=gamma,
            alpha=alpha,
            log_dir=log_dir,
            device=DEVICE
        )
        outcomes = []
        terminated = True
        while q_learning.global_steps < num_steps:
            epsilon = epsilon_scheduler(q_learning.global_steps)
            print(f"\rSteps: {q_learning.global_steps:>5} Buffer: {len(buffer)} epsilon: {epsilon:.2e}", end='')
            if terminated:
                state, mask_actions = convert_to_tensor(env.reset(), device=DEVICE)
                terminated = False

            values = model(state)
            action = epsilon_greedy(
                values=values,
                epsilon=epsilon,
                action_mask=mask_actions
            )[0]
            next_state, reward, terminated, _, _ = env.step(action)
            next_state, next_action_mask = convert_to_tensor(next_state, device=DEVICE)
            buffer.add_experience(
                state=state,
                action=action,
                next_state=next_state,
                next_action_mask=next_action_mask,
                reward=reward,
                terminated=terminated
            )
            state = next_state
            mask_actions = next_action_mask

            if len(buffer) >= min_buffer_size:
                batch = buffer.sample(batch_size)
                q_learning.step(**batch)

                if terminated:
                    outcomes.append(reward)
                    q_learning.logger.add_histogram("WDL", np.array(outcomes))

                if q_learning.global_steps % play_each_n_steps == 0:
                    frames = play_a_game(
                        env=game_env,
                        model=model,
                        device=DEVICE
                    )['frames']

                    frames = np.stack(frames)
                    frames = frames.astype(np.uint8).transpose(0, 3, 1, 2)[None, ...]
                    q_learning.logger.add_video("game", frames, global_step=q_learning.global_steps // play_each_n_steps, fps=1)