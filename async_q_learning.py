import torch
import torch.nn as nn
import torch.multiprocessing as mp
from environment import Connect4Env
from utils import epsilon_greedy, get_model, convert_to_tensor

DEVICE = 'cpu'
PLAYER_0 = 'player_0'
PLAYER_1 = 'player_1'


def worker(
    worker_id:int,
    steps:torch.Tensor,
    steps_to_swap:torch.Tensor,
    model_config:dict,
    global_q_network:nn.Module,
    global_target_network:nn.Module,
    optimizer:torch.optim.Optimizer,
    epsilon:float,
    episodes:int,
    gamma:float = 1.0,
    t_max:int=1,
):
    env = Connect4Env(epsilon=0.1)
    env.load_openning_book("connect_four_solver/7x6.book")

    torch.set_num_threads(1)

    local_net = get_model(model_config)
    local_net.load_state_dict(global_q_network.state_dict())

    for episode in range(episodes):
        state, action_mask = convert_to_tensor(env.reset(), device=DEVICE)
        terminated = False
        episode_rewards = 0

        first_value = local_net(state).detach().tolist()

        while not terminated:
            rewards, state_values = [], []
            for _ in range(t_max):
                values = local_net(state)
                action = epsilon_greedy(values, epsilon=epsilon, action_mask=action_mask)[0]

                next_state, reward, terminated = env.step(action)[PLAYER_0]
                next_state, next_action_mask = convert_to_tensor(next_state, device=DEVICE)

                if not terminated:
                    mini_max_action = env.predict_best_move()
                    next_state, mini_max_reward, terminated = env.step(mini_max_action)[PLAYER_0]
                    next_state, next_action_mask = convert_to_tensor(next_state, device=DEVICE)
                else:
                    mini_max_reward = 0.0

                reward = reward + mini_max_reward
                rewards.append(reward)
                episode_rewards += reward

                state_values.append(values[..., action].reshape(1, 1))

                state = next_state
                action_mask = next_action_mask

                if terminated:
                    break
                steps.add_(1)
                steps_to_swap.subtract_(1)

            with torch.no_grad():
                if not terminated:
                    max_values = global_target_network(next_state)
                    max_action = epsilon_greedy(max_values, 0.0, action_mask)[0]
                    max_value = max_values[..., max_action].reshape(1,)
            R = 0.0 if terminated else max_value.detach()
            returns = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns).unsqueeze(1).to(DEVICE)
            state_values = torch.concat(state_values, dim=0).to(DEVICE)

            td_error = (0.5 * (returns - state_values) ** 2).mean()
            optimizer.zero_grad()
            local_net.zero_grad()
            td_error.backward()

            torch.nn.utils.clip_grad_norm_(local_net.parameters(), 10)

            for local_param, global_param in zip(local_net.parameters(), global_q_network.parameters()):
                global_param.grad = local_param.grad

            optimizer.step()
            local_net.load_state_dict(global_q_network.state_dict())

            # might not run because it might never see 1000 and skip it
            if steps_to_swap < 0:
                global_target_network.load_state_dict(global_q_network.state_dict())
                steps_to_swap.fill_(10000)
                print("Target updated")

        if episode % 100 == 0:
            loss = td_error.detach().cpu().item()
            first_value = [round(x, 2) for x in first_value[0]]
            print(f"worker {worker_id:>2} | episode {episode:>4} | loss: {loss:>4.4e} | Q0: {str(first_value)}")
            

