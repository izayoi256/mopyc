import numpy
from typing import Iterable
from .row import RowViewModel


class LayerViewModel:
    __game_size: int
    __layer: numpy.ndarray
    __commands: numpy.ndarray

    def __init__(self, game_size: int, layer: numpy.ndarray, commands: numpy.ndarray):
        assert layer.shape == commands.shape
        self.__game_size = game_size
        self.__layer = layer
        self.__commands = commands

    @property
    def size(self) -> int:
        return self.__layer.shape[0]

    @property
    def chr(self) -> str:
        return chr(65 + self.__game_size - self.size)

    @property
    def col_chars(self) -> Iterable[str]:
        return map(lambda x: chr(49 + x), range(self.size))

    @property
    def rows(self) -> Iterable[RowViewModel]:
        yield from []
        for i in range(len(self.__layer)):
            yield RowViewModel(
                row_index=i,
                row=self.__layer[i],
                commands=self.__commands[i],
            )
