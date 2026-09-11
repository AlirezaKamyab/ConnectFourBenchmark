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

    env = Connect4Env(epsilon=0.1)
    env.load_openning_book("connect_four_solver/7x6.book")

    game_env = Connect4Env(render_mode='rgb_array')
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
    steps_to_swap_target = config.steps_to_swap_target
    history = None

    for run in range(runs):
        buffer = ExperienceReplay(max_buffer_size, device=DEVICE)
        model = get_model(config.q_network).to(DEVICE)
        q_learning = Q_Learning(
            q_network=model,
            gamma=gamma,
            alpha=alpha,
            log_dir=log_dir,
            steps_to_swap_target=steps_to_swap_target,
            device=DEVICE
        )
        outcomes = []
        terminated = True
        first_move = True
        while q_learning.global_steps < num_steps:
            epsilon = epsilon_scheduler(q_learning.global_steps)
            print(f"\rSteps: {q_learning.global_steps:>5} Buffer: {len(buffer)} epsilon: {epsilon:.2e}", end='')
            if terminated:
                state, action_mask = convert_to_tensor(env.reset(), device=DEVICE)
                terminated = False
                first_move = True

            values = model(state)
            if first_move:
                q_learning.logger.add_scalar(
                    'first_move_expectation', 
                    values.detach().cpu().max().item(), 
                    global_step=q_learning.global_steps)

            action = epsilon_greedy(
                values=values,
                epsilon=epsilon,
                action_mask=action_mask
            )[0]
            after_state, reward, terminated = env.step(action)['player_0']
            first_move = False
            after_state, action_mask = convert_to_tensor(after_state, device=DEVICE)

            if not terminated:
                mini_max_action = env.predict_best_move()
                next_state, next_reward, terminated = env.step(mini_max_action)['player_0']
                next_state, action_mask = convert_to_tensor(next_state, device=DEVICE)
            else:
                next_state = after_state
                next_reward = 0

            reward = next_reward + reward

            buffer.add_experience(
                state=state,
                action=action,
                next_state=next_state,
                next_action_mask=action_mask,
                reward=reward,
                terminated=terminated
            )
            
            state = next_state

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
                    q_learning.logger.flush()
                    torch.save(model, os.path.join(result_path, 'model.pt'))