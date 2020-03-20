import numpy


class CellViewModel:
    __state: int
    __command: str

    def __init__(self, state: int, command: str):
        self.__state = state
        self.__command = command

    def is_first(self) -> bool:
        return self.__state == 1

    def is_second(self) -> bool:
        return self.__state == 2

    def is_legal(self) -> bool:
        return self.__state == 0

    @property
    def command(self) -> str:
        return self.__command
