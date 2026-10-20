import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from env_workers import env_worker
import torch.multiprocessing as mp
from utils import get_model
import json
from types import SimpleNamespace
import numpy as np


def train():
    with open('config.json', "r") as config_file:
        config_file = json.load(config_file)
        config = SimpleNamespace(**config_file)

    epsilon = config.epsilon
    device = config.device
    num_workers = config.num_workers
    tmax = config.tmax
    entropy_coeff = config.entropy_coeff
    critic_coeff = config.critic_coeff
    learning_rate = config.learning_rate
    steps = config.steps
    gamma = config.gamma

    model = get_model(config.actor).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    parent_remote, worker_remote = zip(*[mp.Pipe() for _ in range(num_workers)])
    processes = [
        mp.Process(target=env_worker, args=(worker_remote[i], parent_remote[i], i, epsilon, device))
        for i in range(num_workers)
    ]

    for p in processes:
        p.start()

    for remote in worker_remote:
        remote.close()

    for remote in parent_remote:
        remote.send(('reset', None))
    results = [remote.recv() for remote in parent_remote]
    states, action_masks = zip(*results)
    states = torch.vstack(states)
    action_masks = np.vstack(action_masks)
    all_outcomes = []
    initial_state = states

    for step in range(steps):
        rewards, state_values, log_probs, entropies, terminates = [], [], [], [], []

        for t in range(tmax):
            logits, values = model(states)

            # prepare the mask
            action_masks = torch.tensor(action_masks, dtype=torch.bool, device=device)
            action_masks = ~action_masks

            masked_logits = torch.masked_fill(logits, action_masks, value=-torch.inf)
            probs = F.softmax(masked_logits, dim=-1)

            dist = Categorical(probs=probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).unsqueeze(1)
            entropy = dist.entropy().unsqueeze(1)

            actions = actions.cpu().detach().tolist()

            for remote, action in zip(parent_remote, actions):
                remote.send(('step', action))


            results = [remote.recv() for remote in parent_remote]
            next_states, action_masks, reward, terminated, outcomes  = zip(*results)
            all_outcomes.extend([x for x in outcomes if x in ['win', 'lose', 'draw']])
            next_states = torch.vstack(next_states)
            action_masks = np.vstack(action_masks)
            reward = np.vstack(reward)
            terminated = np.vstack(terminated)

            rewards.append(reward)
            state_values.append(values)
            log_probs.append(log_prob)
            entropies.append(entropy)
            terminates.append(terminated)

            states = next_states


        with torch.no_grad():
            _, next_values = model(next_states)

        R = next_values.cpu().numpy()
        returns = []
        for i in reversed(range(len(rewards))):
            r = rewards[i]
            term = terminates[i].astype(np.float32)
            R = r + (1 - term) * gamma * R
            returns.insert(0, R)

        returns = np.concat(returns, axis=1)
        returns = torch.from_numpy(returns).to(device=device)
        values = torch.concat(state_values, dim=-1).to(device=device)
        log_probs = torch.concat(log_probs, dim=-1).to(device=device)
        entropies = torch.concat(entropies, dim=-1).to(device=device)

        advantage = returns - values
        critic_loss = 0.5 * torch.square(advantage).mean()
        actor_loss = -(advantage.detach() * log_probs).mean() - entropy_coeff * entropies.mean()

        optimizer.zero_grad()
        model.zero_grad()
        loss = actor_loss + critic_coeff * critic_loss
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        optimizer.step()

        if step % 100 == 0:
            total_games = len(all_outcomes)
            mapper = {'win':1, 'lose':-1, 'draw':0}
            all_outcomes = [mapper[x] for x in all_outcomes]
            all_outcomes = np.array(all_outcomes)
            win_rate = (all_outcomes == 1).astype(np.float32).sum() / total_games
            lose_rate = (all_outcomes == -1).astype(np.float32).sum() / total_games
            draw_rate = (all_outcomes == 0).astype(np.float32).sum() / total_games

            all_outcomes = []

            with torch.no_grad():
                pi0 = torch.nn.functional.softmax(model(initial_state)[0], dim=-1)[0].cpu()
                pi0 = [round(x, 2) for x in pi0.tolist()]

            print(f"Step {step:>5} | win_rate {win_rate:.2f} | lose_rate {lose_rate:.2f} | draw_rate {draw_rate:.2f} | actor_loss {actor_loss.item():3.3e} | critic_loss {critic_loss.item():3.3e} | grad_norm: {grad_norm.item():>2.2f} | pi_0: {str(pi0)}")
            torch.save(model, 'checkpoint.pt')


if __name__ == '__main__':
    train()