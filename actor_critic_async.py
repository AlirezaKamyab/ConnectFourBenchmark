import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Connect4Env
from models.mlp import MLP
from models.mlp_actor import ActorMLP
from utils import get_model, convert_to_tensor

PLAYER_0 = 'player_0'
PLAYER_1 = 'player_1'
LOG_EVERY = 100


def worker(
    worker_id:int,
    config:dict,
    global_actor_critic:nn.Module,
    optimizer:torch.optim.Optimizer,
    gamma:float,
    episodes:int,
    t_max:int=1,
    entropy_coef:float=0.0
):
    torch.set_num_threads(1)

    local_actor_critic = get_model(config)
    local_actor_critic.load_state_dict(global_actor_critic.state_dict())

    env = Connect4Env(epsilon=0.1)
    env.load_openning_book('connect_four_solver/7x6.book')
    wins, loses, draws = 0, 0, 0
    length = 0

    for episode in range(episodes):
        # if episode == 2000:
        #     epsilon = 0.3
        #     env = Connect4Env(epsilon=epsilon)
        #     env.load_openning_book('connect_four_solver/7x6.book')
        # if episode == 4000:
        #     epsilon = 0.2
        #     env = Connect4Env(epsilon=epsilon)
        #     env.load_openning_book('connect_four_solver/7x6.book')
        # if episode == 6000:
        #     epsilon = 0.1
        #     env = Connect4Env(epsilon=epsilon)
        #     env.load_openning_book('connect_four_solver/7x6.book')

        state = env.reset()
        state, action_mask = convert_to_tensor(state)

        pi0 = None

        terminated = False
        while not terminated:
            values, rewards, entropies, log_probs = [], [], [], []

            for _ in range(t_max):
                logits, value = local_actor_critic(state)

                # prepare mask
                action_mask = torch.tensor(action_mask, dtype=torch.bool)
                action_mask = ~action_mask

                # mask the logits
                masked_logits = logits.masked_fill(action_mask, -torch.inf)
                probs = F.softmax(masked_logits, dim=-1)

                if pi0 is None:
                    pi0 = probs.detach()[0].numpy()
                
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                np_action = action.detach().numpy()[0]
                next_state, reward, terminated = env.step(np_action)[PLAYER_0]
                next_state, action_mask = convert_to_tensor(next_state)

                if not terminated:
                    best_action = env.predict_best_move()
                    next_state, next_reward, terminated = env.step(best_action)[PLAYER_0]
                    next_state, action_mask = convert_to_tensor(next_state)
                else:
                    next_reward = 0.0

                reward = reward + next_reward
                if terminated and reward == 1:
                    wins += 1
                elif terminated and reward == 0:
                    draws += 1
                elif terminated and reward == -1:
                    loses += 1

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

                state = next_state

                length += 1
                if terminated:
                    break

            R = 0.0 if terminated else local_actor_critic(state)[1].item()
            returns = []

            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.concat(values, dim=0).squeeze(-1)
            log_probs = torch.concat(log_probs, dim=0)
            entropies = torch.concat(entropies, dim=0)

            advantages = returns - values

            # critic update
            local_actor_critic.zero_grad()
            optimizer.zero_grad()

            critic_loss = advantages.square().mean()
            actor_loss = -(advantages.detach() * log_probs).mean() - entropy_coef * entropies.mean()
            loss = actor_loss + critic_loss
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(local_actor_critic.parameters(), 1)

            for local_param, global_param in zip(local_actor_critic.parameters(), global_actor_critic.parameters()):
                if global_param.grad is not None:
                    global_param.grad.copy_(local_param.grad)
                else:
                    global_param._grad = local_param.grad

            optimizer.step()

            local_actor_critic.load_state_dict(global_actor_critic.state_dict())

        if episode % LOG_EVERY == 0:
            pi0 = [round(x, 2) for x in pi0.tolist()]
            win_rate = wins / LOG_EVERY
            lose_rate = loses / LOG_EVERY
            draw_rate = draws / LOG_EVERY
            advantage_log = advantages.detach().mean().item()
            mean_length = length / LOG_EVERY
            length = 0

            wins, loses, draws = 0, 0, 0
            print(f"worker: {worker_id} | episode {episode:>4} | length {mean_length:>3.0f} | win_rate {win_rate:.2f} | lose_rate {lose_rate:.2f} | draw_rate {draw_rate:.2f} | advantage {advantage_log:.3e} | pi_0: {str(pi0)}")