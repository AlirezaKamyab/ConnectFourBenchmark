from connect_four_solver import connect4 as connect4solver
from pettingzoo.classic import connect_four_v3
from gymnasium import Env
import numpy as np


class Connect4Env(Env):
    def __init__(
      self,
      render_mode:str=None,
      weak_solver:bool=False,
      epsilon:float=0.0
    ):
        super(Connect4Env, self).__init__()
        self.render_mode = render_mode
        self.env = connect_four_v3.env(render_mode=render_mode)
        self.solver = connect4solver.Solver(weak_solver)
        self._sequence = ""
        self.epsilon = epsilon

    def load_openning_book(self, openning_book_path:str):
        self.solver.load_book(openning_book_path)
        print("Openning Book has been loaded!")

    def evaluate(self) -> int:
        return self.solver.solve(self._sequence)

    def analyze(self) -> list:
        values = self.solver.analyze(self._sequence)
        return values

    def get_all_best_actions(self):
        values = np.array(self.analyze())
        max_value = np.max(values)
        return np.argwhere(values == max_value)[:, 0].tolist()

    def predict_best_move(self) -> int:
        values = self.analyze()
        values = np.array(values)
        max_values = np.argwhere(values == values.max())[:, 0]
        best = int(np.random.choice(max_values))

        if self.epsilon > np.random.rand():
            mask = self.env.last()[0]['action_mask']
            values = np.argwhere(mask)[:, 0]
            return np.random.choice(values)
        return best

    def get_feedback(self, agent:str='player_0'):
        obs = self.env.observe(agent=agent)
        reward = self.env.rewards[agent]
        terminated = self.env.terminations
        terminated = terminated['player_0'] or terminated['player_1']
        return obs, reward, terminated

    def reset(self, seed:int=None):
        self.env.reset(seed=seed)
        self._sequence = ""
        return self.env.last()[0]

    def step(self, action:int):
        self.env.step(action)
        self._sequence += f"{action+1}"
        feedbacks = {
            'player_0':self.get_feedback('player_0'),
            'player_1':self.get_feedback('player_1')
        }
        return feedbacks

    def render(self):
        return self.env.render()

    def get_player(self) -> str:
        return self.env.agent_selection
        