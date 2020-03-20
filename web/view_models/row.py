import numpy
from typing import Iterable
from .cell import CellViewModel


class RowViewModel:
    __row_index: int
    __row: numpy.ndarray
    __commands: numpy.ndarray

    def __init__(self, row_index: int, row: numpy.ndarray, commands: numpy.ndarray):
        assert row.shape == commands.shape
        self.__row_index = row_index
        self.__row = row
        self.__commands = commands

    @property
    def size(self) -> int:
        return self.__row.shape[0]

    @property
    def chr(self) -> str:
        return chr(97 + self.__row_index)

    @property
    def cells(self) -> Iterable[CellViewModel]:
        yield from []
        for i in range(len(self.__row)):
            yield CellViewModel(
                state=self.__row[i],
                command=self.__commands[i],
            )
