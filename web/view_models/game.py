import numpy
from typing import Iterable, List
from .layer import LayerViewModel
from .command import CommandViewModel
from ai.gym.envs.mosaic import MosaicObservation


class GameViewModel:
    __observation: MosaicObservation
    __agent_available: bool

    def __init__(self, observation: MosaicObservation, agent_available: bool):
        self.__observation = observation
        self.__agent_available = agent_available

    @property
    def size(self) -> int:
        return self.__observation.game.size

    @property
    def is_over(self) -> bool:
        return self.__observation.game.is_over()

    @property
    def first_wins(self) -> bool:
        return self.__observation.game.first_wins()

    @property
    def second_wins(self) -> bool:
        return self.__observation.game.second_wins()

    @property
    def is_first_turn(self) -> bool:
        return self.__observation.game.is_first_turn()

    @property
    def is_second_turn(self) -> bool:
        return self.__observation.game.is_second_turn()

    @property
    def first_score(self) -> int:
        return self.__observation.game.first_score()

    @property
    def second_score(self) -> int:
        return self.__observation.game.second_score()

    @property
    def first_pretty_moves(self) -> List[str]:
        return list(reversed(self.__observation.game.first_pretty_moves()))

    @property
    def second_pretty_moves(self) -> List[str]:
        return list(reversed(self.__observation.game.second_pretty_moves()))

    @property
    def layers(self) -> Iterable[LayerViewModel]:
        yield from []
        layer_size = self.size
        array = self.__observation.single_array
        pretty_moves = numpy.array([self.__observation.game.move_to_pretty(i) for i in range(len(array))])
        while True:
            if layer_size < 1:
                break
            start = sum([i ** 2 for i in range(layer_size)])
            end = sum([i ** 2 for i in range(1, layer_size + 1)])
            yield LayerViewModel(
                game_size=self.size,
                layer=array[start:end].reshape(layer_size, layer_size),
                commands=pretty_moves[start:end].reshape(layer_size, layer_size),
            )
            layer_size -= 1

    @property
    def commands(self) -> Iterable[CommandViewModel]:
        yield from []
        if self.__agent_available and not self.__observation.game.is_over():
            yield CommandViewModel('Make AI move', 'agent')
        if self.__observation.game.moves_made() > 0:
            yield CommandViewModel('Undo', 'undo')
        yield CommandViewModel('Exit', 'exit')
