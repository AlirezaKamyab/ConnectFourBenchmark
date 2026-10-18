import torch.multiprocessing as mp
from environment import Connect4Env
from utils import convert_to_tensor

PLAYER_0 = 'player_0'


def env_worker(remote, parent_remote, worker_id, epsilon=0.1, device='cpu'):
    parent_remote.close()
    env = Connect4Env(epsilon=epsilon)
    env.load_openning_book("connect_four_solver/7x6.book")

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                action = data
                obs, reward, terminated = env.step(action)[PLAYER_0]
                state, action_mask = convert_to_tensor(obs, device=device)

                if not terminated:
                    obs, next_reward, terminated = env.step(env.predict_best_move())[PLAYER_0]
                    state, action_mask = convert_to_tensor(obs, device=device)
                else:
                    next_reward = 0

                reward = reward + next_reward


                if terminated:
                    state = env.reset()
                    state, action_mask = convert_to_tensor(state, device=device)
                    if reward == 1:
                        outcome = 'win'
                    elif reward == 0:
                        outcome = 'draw'
                    else:
                        outcome = 'lose'
                else:
                    outcome = 'nothing'

                remote.send((state, action_mask, reward, terminated, outcome)) 
                    

            elif cmd == 'reset':
                state = env.reset()
                state, action_mask = convert_to_tensor(state, device=device)
                remote.send((state, action_mask))
            elif cmd == 'close':
                remote.close()
                break

        except EOFError:
            break


def main():
    parent_remotes, worker_remotes = zip(*[mp.Pipe() for _ in range(8)])
    processes = [
        mp.Process(target=env_worker, args=(worker_remotes[i], parent_remotes[i], i))
        for i in range(len(worker_remotes))
    ]

    for p in processes:
        p.start()
    for remote in worker_remotes: remote.close()

    for remote in parent_remotes:
        remote.send(('reset', None))

    results = [remote.recv() for remote in parent_remotes]
    print(results)

    for remote in parent_remotes:
        remote.send(('close', None))

if __name__ == '__main__':
    main()