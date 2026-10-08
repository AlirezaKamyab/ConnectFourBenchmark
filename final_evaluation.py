import torch
from utils import final_evaluation_actor
from environment import Connect4Env
from argparse import ArgumentParser



def main(args):
    env = Connect4Env(render_mode="rgb_array", epsilon=0)
    env.load_openning_book("connect_four_solver/7x6.book")
    model = torch.load(args.model_path, weights_only=False)

    for epsilon in [0.0, 0.01, 0.05, 0.1, 1.0]:
        print(f"{epsilon}-opponent")
        env.epsilon = epsilon
        outputs = final_evaluation_actor(env, model, runs=1000)
        print('Win-rate', outputs['win_rate'])
        print('Draw-rate', outputs['draw_rate'])
        print('Lose-rate' ,outputs['lose_rate'])
        print('Mean-optimal-rate', outputs['mean_optimal_rate'])
        print('-'*10)


if __name__ == '__main__':
    argparse = ArgumentParser(description="no desc")
    argparse.add_argument("--model-path", '-m', required=True)
    args = argparse.parse_args()
    main(args)