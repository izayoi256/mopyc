import random
from core import Game, Player


class RandomPlayer(Player):
    def make_move(self, game: Game):
        return random.choice(list(game.legal_moves()))
