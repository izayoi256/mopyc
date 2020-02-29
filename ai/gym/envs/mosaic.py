import numpy
import random
from typing import Callable, NoReturn, Optional, Tuple, Union
from ai.random import RandomPlayer
from core import Game
from gym import Env, spaces

Action = Union[int, numpy.integer]


class MosaicObservation:
    __legal_box: numpy.ndarray
    __player_box: numpy.ndarray
    __opponent_box: numpy.ndarray
    __array: Optional[numpy.ndarray]
    __game: Game

    def __init__(self, game: Game, legal_box: numpy.ndarray, player_box: numpy.ndarray, opponent_box: numpy.ndarray):
        self.__game = game
        self.__legal_box = legal_box.copy()
        self.__player_box = player_box.copy()
        self.__opponent_box = opponent_box.copy()
        self.__array = None

    @property
    def game(self):
        return self.__game

    @property
    def legal_box(self) -> numpy.ndarray:
        return self.__legal_box

    @property
    def legal_actions(self) -> numpy.ndarray:
        return self.__legal_box.nonzero()[0]

    @property
    def player_box(self) -> numpy.ndarray:
        return self.__player_box

    @property
    def opponent_box(self) -> numpy.ndarray:
        return self.__opponent_box

    @property
    def array(self) -> numpy.ndarray:
        if self.__array is None:
            self.__array = numpy.array([self.__legal_box, self.__player_box, self.__opponent_box])
        return self.__array

    @property
    def single_array(self) -> numpy.ndarray:
        first_box = self.array[1] if self.game.is_first_turn() else self.array[2]
        second_box = self.array[2] if self.game.is_first_turn() else self.array[1]
        return self.array[0] - 1 + first_box * 2 + second_box * 3


# MosaicObservation = namedtuple('MosaicObservation', ('array', 'game'))
Policy = Callable[[MosaicObservation], Action]

DTYPE = numpy.int8
random_player = RandomPlayer()


def cast_action(action: Action):
    return action.item() if issubclass(type(action), numpy.integer) else action


class RandomPolicy(Policy):
    def __call__(self, observation: MosaicObservation):
        return random.choice(observation.legal_actions)


random_policy = RandomPolicy()


class MosaicEnv(Env):
    __size: int
    __actions: int
    __is_first_player: bool
    __game: Game
    __opponent_policy: Policy
    __done_on_illegal_move: bool

    def __init__(self, size: int, done_on_illegal_move: bool):
        self.__size = size
        self.__actions = sum([(i + 1) ** 2 for i in range(self.__size)])
        self.__opponent_policy = random_policy
        self.__done_on_illegal_move = done_on_illegal_move

        self.action_space = spaces.Discrete(self.__actions)
        self.observation_space = spaces.Box(0, 1, (self.__actions, 3), dtype=DTYPE)

    def reset(self,
              is_first_player: bool = True,
              opponent_policy: Optional[Policy] = None,
              transform: bool = True,
              reset_transformation: bool = False,
              ) -> MosaicObservation:
        self.__is_first_player = is_first_player
        self.__opponent_policy = opponent_policy or random_policy
        self.__game = Game(self.__size)
        if not self.__is_first_player:
            self.__make_opponent_move()
        return self.observe(transform=transform, reset_transformation=reset_transformation)

    def step(self, action: Action) -> Tuple[MosaicObservation, float, bool, dict]:

        info = {
            'illegal_action': False,
            'win': False,
        }

        int_action = cast_action(action)

        if not self.__game.is_legal_move(int_action):
            reward = self.__illegal_move_reward()
            done = self.__done_on_illegal_move
            info['illegal_action'] = True
        else:

            self.__game.move(int_action)
            reward = self.__legal_move_reward()
            done = self.__game.is_over()

            if not done:
                self.__make_opponent_move()
                done = self.__game.is_over()

            if done:
                win = self.__player_wins()
                reward += self.__win_reward() if win else self.__lose_reward()
                info['win'] = win

        return self.observe(), reward, done, info

    def render(self, mode: str = 'human') -> NoReturn:
        print(self.__game.format())

    def sample(self) -> Action:
        return random_player.make_move(self.__game)

    def __legal_move_reward(self) -> float:
        return 0.0

    def __illegal_move_reward(self) -> float:
        return -1.0

    def __lose_reward(self) -> float:
        return -0.5

    def __win_reward(self) -> float:
        return 0.5

    def __player_wins(self) -> bool:
        return not (self.__game.first_wins() ^ self.__is_first_player)

    def __player_score(self) -> int:
        return self.__game.first_score() if self.__is_first_player else self.__game.second_score()

    def __opponent_score(self) -> int:
        return self.__game.second_score() if self.__is_first_player else self.__game.first_score()

    def __player_moves(self) -> int:
        return len(list(self.__game.first_moves() if self.__is_first_player else self.__game.second_moves()))

    def __opponent_moves(self) -> int:
        return len(list(self.__game.second_moves() if self.__is_first_player else self.__game.first_moves()))

    def __make_opponent_move(self) -> NoReturn:
        opponent_observation = self.observe()
        action = self.__opponent_policy(opponent_observation)
        action = cast_action(action)
        self.__game.move(action)

    def __board_to_box(self, board: str) -> numpy.ndarray:
        int_board = int(board, 2)
        return numpy.array([1 if (int_board >> i) & 1 == 1 else 0 for i in range(self.__actions)], dtype=DTYPE)

    def observe(self, transform: bool = True, reset_transformation: bool = False) -> MosaicObservation:
        if reset_transformation:
            transform = False
            self.__game.reset_transformation()
        if transform:
            self.__game.transform()
        return MosaicObservation(
            game=self.__game,
            legal_box=self.__board_to_box(self.__game.legal_board()),
            player_box=self.__board_to_box(self.__game.player_board()),
            opponent_box=self.__board_to_box(self.__game.opponent_board()),
        )
