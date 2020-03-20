class CommandViewModel:
    __name: str
    __command: str

    def __init__(self, name: str, command: str):
        self.__name = name
        self.__command = command

    def __str__(self) -> str:
        return self.__command

    @property
    def command(self) -> str:
        return self.__command

    @property
    def name(self) -> str:
        return self.__name
