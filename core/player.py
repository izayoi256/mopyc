from abc import ABC, abstractmethod
from . import Game


class Player(ABC):

    @abstractmethod
    def make_move(self, game: Game) -> int:
        pass
