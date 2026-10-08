import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Connect4Env
from utils import get_model, convert_to_tensor, change_learning_rate, linear_epsilon_scheduler
from torch.utils.tensorboard import SummaryWriter

PLAYER_0 = 'player_0'
PLAYER_1 = 'player_1'
LOG_EVERY = 100

logger = SummaryWriter('./logs')

def worker(
    worker_id:int,
    config:dict,
    global_actor_critic:nn.Module,
    optimizer:torch.optim.Optimizer,
    gamma:float,
    episodes:int,
    global_steps:torch.Tensor,
    t_max:int=1,
    entropy_coef:float=0.0,
    critic_coef:float=0.0,
):
    torch.set_num_threads(1)

    local_actor_critic = get_model(config)
    local_actor_critic.load_state_dict(global_actor_critic.state_dict())
    lr_scheduler = linear_epsilon_scheduler(1e-4, 1e-5, 10_000_000)

    env = Connect4Env(epsilon=0.1)
    env.load_openning_book('connect_four_solver/7x6.book')
    wins, loses, draws = 0, 0, 0
    length = 0
    worker_steps = 0

    for episode in range(1, episodes + 1):
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
            loss = actor_loss + critic_coef * critic_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(local_actor_critic.parameters(), 40.0)
            change_learning_rate(optimizer, lr_scheduler(global_steps))
            

            for local_param, global_param in zip(local_actor_critic.parameters(), global_actor_critic.parameters()):
                if global_param.grad is not None:
                    global_param.grad.copy_(local_param.grad)
                else:
                    global_param._grad = local_param.grad

            optimizer.step()
            worker_steps += 1
            global_steps.add_(1.0)

            advantage_log = advantages.detach().mean().item()
            actor_loss_log = actor_loss.detach().mean().item()
            critic_loss_log = critic_loss.detach().mean().item()

            local_actor_critic.load_state_dict(global_actor_critic.state_dict())

            # Log
            if worker_id == 0:
                logger.add_scalar(f"worker_{worker_id}/advantage", advantage_log, global_step=worker_steps)
                logger.add_scalars(f"worker_{worker_id}/loss", {
                    'critic_loss':critic_loss_log,
                    'actor_loss':actor_loss_log,
                    'loss': loss.detach().mean().item()
                }, global_step=worker_steps)
                logger.add_scalar(f'worker_{worker_id}/grad_norm', grad_norm.item(), global_step=worker_steps)


        if episode % LOG_EVERY == 0:
            pi0 = [round(x, 2) for x in pi0.tolist()]
            win_rate = wins / LOG_EVERY
            lose_rate = loses / LOG_EVERY
            draw_rate = draws / LOG_EVERY
            mean_length = length / LOG_EVERY
            length = 0

            wins, loses, draws = 0, 0, 0

            if worker_id == 0:
                logger.add_scalars(f"worker_{worker_id}/wdl", {
                    'win_rate':win_rate,
                    'lose_rate':lose_rate,
                    'draw_rate':draw_rate
                }, global_step=episode//LOG_EVERY)

            print(f"worker: {worker_id:>2} | episode {episode:>4} | length {mean_length:>3.0f} | win_rate {win_rate:.2f} | lose_rate {lose_rate:.2f} | draw_rate {draw_rate:.2f} | actor_loss {actor_loss_log:3.3e} | critic_loss {critic_loss_log:3.3e} | lr {lr_scheduler(global_steps):3.3e} | pi_0: {str(pi0)}")
