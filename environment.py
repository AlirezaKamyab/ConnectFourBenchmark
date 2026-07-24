from connect_four_solver import connect4 as connect4solver
from pettingzoo.classic import connect_four_v3
from gymnasium import Env
import numpy as np


class Connect4Env(Env):
    def __init__(
      self,
      render_mode:str=None,
      weak_solver:bool=False,
      computer_player:str="player_1",
      simple_reward:bool=True,
      temperature:float=0.0
    ):
        super(Connect4Env, self).__init__()
        self.render_mode = self.render_mode
        self.env = connect_four_v3.env(render_mode=render_mode)
        self.solver = connect4solver.Solver(weak_solver)
        self._sequence = ""
        self.computer_player = computer_player
        self.simple_reward = simple_reward
        self.actions_performed = [0, 0]
        self.temperature = temperature

    def load_openning_book(self, openning_book_path:str):
        self.solver.load_book(openning_book_path)
        print("Openning Book has been loaded!")

    def evaluate(self) -> int:
        return self.solver.solve(self._sequence)

    def analyze(self) -> list:
        values = self.solver.analyze(self._sequence)
        if self.temperature != 0.0:
            values = np.array(values, dtype=np.float32)
            values = values - values.max()
            exp_values = np.exp(values / self.temperature)
            values = exp_values / np.sum(exp_values)

        return values

    def get_all_best_actions(self):
        values = np.array(self.analyze())
        max_value = np.max(values)
        return np.argwhere(values == max_value)[:, 0].tolist()

    def predict_best_move(self) -> int:
        values = self.analyze()
        values = np.array(values)
        if self.temperature == 0.0:
            return int(np.argmax(values))

        return np.random.choice(len(values), p=values)

    def reset(self, seed:int=None):
        self.env.reset(seed=seed)
        self._sequence = ""
        self.actions_performed = [0, 0]

        if self.get_player() == self.computer_player:
            return self.step(self.predict_best_move())[0]
        return self.env.last()[0]

    def step(self, action:int):
        if self.get_player() == 'player_0':
            self.actions_performed[0] += 1
        else:
            self.actions_performed[1] += 1

        self.env.step(action)
        self._sequence += f"{action+1}"
        terminated = self.env.last()[1]
        if self.computer_player == self.get_player() and not terminated:
            return self.step(self.predict_best_move())
        if self.simple_reward:
            return self.env.last()
        else:
            obs, reward, terminated, truncated, info = self.env.last()
            if self.get_player() == 'player_0':
                reward = reward * (21 - self.actions_performed[0])
                return obs, reward, terminated, truncated, info
            else:
                reward = reward * (21 - self.actions_performed[1])
                return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def get_player(self) -> str:
        return self.env.agent_selection
        