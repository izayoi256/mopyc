from ctypes import *
import os
from typing import Callable, Dict, Iterable, NoReturn, Optional


def load_library():
    is_windows = os.name == 'nt'
    ext = 'dll' if is_windows else 'so'
    path = os.path.join(os.path.dirname(__file__), '..', f'libmosaicgame.{ext}')
    lib = cdll.LoadLibrary(path)
    lib.create.argtypes = (c_uint,)
    lib.create.restype = c_void_p
    lib.destroy.argtypes = (c_void_p,)
    lib.destroy.restype = None
    lib.legalBoard.argtypes = (c_void_p, c_char_p)
    lib.legalBoard.restype = None
    lib.firstBoard.argtypes = (c_void_p, c_char_p)
    lib.firstBoard.restype = None
    lib.secondBoard.argtypes = (c_void_p, c_char_p)
    lib.secondBoard.restype = None
    lib.playerBoard.argtypes = (c_void_p, c_char_p)
    lib.playerBoard.restype = None
    lib.opponentBoard.argtypes = (c_void_p, c_char_p)
    lib.opponentBoard.restype = None
    lib.neutralBoard.argtypes = (c_void_p, c_char_p)
    lib.neutralBoard.restype = None
    lib.isOver.argtypes = (c_void_p,)
    lib.isOver.restype = c_bool
    lib.isFirstTurn.argtypes = (c_void_p,)
    lib.isFirstTurn.restype = c_bool
    lib.isSecondTurn.argtypes = (c_void_p,)
    lib.isSecondTurn.restype = c_bool
    lib.firstWins.argtypes = (c_void_p,)
    lib.firstWins.restype = c_bool
    lib.secondWins.argtypes = (c_void_p,)
    lib.secondWins.restype = c_bool
    lib.isLegalMove.argtypes = (c_void_p, c_uint16)
    lib.isLegalMove.restype = c_bool
    lib.move.argtypes = (c_void_p, c_uint16)
    lib.move.restype = None
    lib.undo.argtypes = (c_void_p, c_uint16)
    lib.undo.restype = None
    lib.movesMade.argtypes = (c_void_p,)
    lib.movesMade.restype = c_uint16
    lib.getMove.argtypes = (c_void_p, c_uint16)
    lib.getMove.restype = c_uint16
    lib.getTransformedMove.argtypes = (c_void_p, c_uint16)
    lib.getTransformedMove.restype = c_uint16
    lib.flipVertical.argtypes = (c_void_p,)
    lib.flipVertical.restype = None
    lib.mirrorHorizontal.argtypes = (c_void_p,)
    lib.mirrorHorizontal.restype = None
    lib.flipDiagonal.argtypes = (c_void_p,)
    lib.flipDiagonal.restype = None
    lib.rotate90.argtypes = (c_void_p,)
    lib.rotate90.restype = None
    lib.rotate180.argtypes = (c_void_p,)
    lib.rotate180.restype = None
    lib.rotate270.argtypes = (c_void_p,)
    lib.rotate270.restype = None
    lib.flipVertical.argtypes = (c_void_p,)
    lib.flipVertical.restype = None
    lib.transform.argtypes = (c_void_p,)
    lib.transform.restype = None
    lib.resetTransformation.argtypes = (c_void_p,)
    lib.resetTransformation.restype = None
    return lib


class PrettyMoveException(Exception):
    pass


class Game:
    __pointer: int
    __size: int
    __first_symbol: str
    __second_symbol: str
    __neutral_symbol: str
    __legal_symbol: str
    __empty_symbol: str
    __x_char: Callable[[int, int], str]
    __y_char: Callable[[int, int], str]
    __z_char: Callable[[int, int], str]
    __PRETTY_MOVES: Optional[Dict[str, int]]
    __MOVES_PRETTY: Optional[Dict[int, str]]

    __LIB = None

    def __init__(self, size: int, first_symbol: str = 'O', second_symbol: str = 'X', neutral_symbol: str = '@',
                 legal_symbol: str = '.', empty_symbol: str = ' ',
                 x_char: Callable[[int, int], str] = lambda layer_size, x: chr(48 + x),
                 y_char: Callable[[int, int], str] = lambda size, y: chr(65 + size - y),
                 z_char: Callable[[int, int], str] = lambda layer_size, z: chr(96 + z),
                 ):
        self.__pointer = self.__lib.create(size)
        self.__size = size
        self.__first_symbol = first_symbol
        self.__second_symbol = second_symbol
        self.__neutral_symbol = neutral_symbol
        self.__legal_symbol = legal_symbol
        self.__empty_symbol = empty_symbol
        self.__x_char = x_char
        self.__y_char = y_char
        self.__z_char = z_char
        self.__PRETTY_MOVES = None
        self.__MOVES_PRETTY = None

    @property
    def __lib(self):
        if Game.__LIB is None:
            Game.__LIB = load_library()
        return Game.__LIB

    @property
    def size(self):
        return self.__size

    @property
    def __pretty_moves(self) -> Dict[str, int]:
        if self.__PRETTY_MOVES is None:
            action = 0
            self.__PRETTY_MOVES = {}
            for y in range(1, self.__size + 1):
                layer_size = y
                for z in range(1, y + 1):
                    for x in range(1, y + 1):
                        k = self.__y_char(self.__size, y) + self.__x_char(layer_size, x) + self.__z_char(layer_size, z)
                        self.__PRETTY_MOVES[k] = action
                        action += 1
        return self.__PRETTY_MOVES

    @property
    def __moves_pretty(self) -> Dict[int, str]:
        if self.__MOVES_PRETTY is None:
            self.__MOVES_PRETTY = {v: k for k, v in self.__pretty_moves.items()}
        return self.__MOVES_PRETTY

    def legal_moves(self) -> Iterable[int]:
        buffer = create_string_buffer(140)
        self.__lib.legalBoard(self.__pointer, buffer)
        tuples = [(index, bit) for index, bit in enumerate(list(reversed(buffer.value.decode())))]
        filtered = filter(lambda t: t[1] == '1', tuples)
        return map(lambda f: f[0], filtered)

    def pretty_legal_moves(self) -> Iterable[str]:
        return map(lambda m: self.move_to_pretty(m), self.legal_moves())

    def first_board(self) -> str:
        buffer = create_string_buffer(140)
        self.__lib.firstBoard(self.__pointer, buffer)
        return buffer.value.decode()

    def second_board(self) -> str:
        buffer = create_string_buffer(140)
        self.__lib.secondBoard(self.__pointer, buffer)
        return buffer.value.decode()

    def player_board(self) -> str:
        buffer = create_string_buffer(140)
        self.__lib.playerBoard(self.__pointer, buffer)
        return buffer.value.decode()

    def opponent_board(self) -> str:
        buffer = create_string_buffer(140)
        self.__lib.opponentBoard(self.__pointer, buffer)
        return buffer.value.decode()

    def legal_board(self) -> str:
        buffer = create_string_buffer(140)
        self.__lib.legalBoard(self.__pointer, buffer)
        return buffer.value.decode()

    def neutral_board(self) -> str:
        buffer = create_string_buffer(140)
        self.__lib.neutralBoard(self.__pointer, buffer)
        return buffer.value.decode()

    def move(self, move: int) -> NoReturn:
        self.__lib.move(self.__pointer, move)

    def pretty_move(self, pretty_move: str) -> NoReturn:
        self.move(self.pretty_to_move(pretty_move))

    def moves_made(self) -> int:
        return c_int(self.__lib.movesMade(self.__pointer)).value

    def get_move(self, move_index: int) -> int:
        return c_int(self.__lib.getMove(self.__pointer, move_index)).value

    def get_pretty_move(self, move_index: int) -> str:
        return self.move_to_pretty(self.get_move(move_index))

    def moves(self) -> Iterable[int]:
        return map(lambda i: self.get_move(i), range(self.moves_made()))

    def pretty_moves(self) -> Iterable[str]:
        return map(lambda i: self.get_pretty_move(i), range(self.moves_made()))

    def first_moves(self) -> Iterable[int]:
        return list(self.moves())[0::2]

    def first_pretty_moves(self) -> Iterable[str]:
        return list(self.pretty_moves())[0::2]

    def second_moves(self) -> Iterable[int]:
        return list(self.moves())[1::2]

    def second_pretty_moves(self) -> Iterable[str]:
        return list(self.pretty_moves())[1::2]

    def get_transformed_move(self, move_index: int) -> int:
        return c_int(self.__lib.getTransformedMove(self.__pointer, move_index)).value

    def get_transformed_pretty_move(self, move_index: int) -> str:
        return self.move_to_pretty(self.get_transformed_move(move_index))

    def transformed_moves(self) -> Iterable[int]:
        return map(lambda i: self.get_transformed_move(i), range(self.moves_made()))

    def transformed_pretty_moves(self) -> Iterable[str]:
        return map(lambda i: self.get_transformed_pretty_move(i), range(self.moves_made()))

    def first_transformed_moves(self) -> Iterable[int]:
        return list(self.transformed_moves())[0::2]

    def first_transformed_pretty_moves(self) -> Iterable[str]:
        return list(self.transformed_pretty_moves())[0::2]

    def second_transformed_moves(self) -> Iterable[int]:
        return list(self.transformed_moves())[1::2]

    def second_transformed_pretty_moves(self) -> Iterable[str]:
        return list(self.transformed_pretty_moves())[1::2]

    def undo(self, moves: int = 1) -> NoReturn:
        self.__lib.undo(self.__pointer, moves)

    def is_first_turn(self) -> bool:
        return c_bool(self.__lib.isFirstTurn(self.__pointer)).value

    def is_second_turn(self) -> bool:
        return c_bool(self.__lib.isSecondTurn(self.__pointer)).value

    def is_over(self) -> bool:
        return c_bool(self.__lib.isOver(self.__pointer)).value

    def is_legal_move(self, move: int) -> bool:
        return c_bool(self.__lib.isLegalMove(self.__pointer, move)).value

    def is_legal_pretty_move(self, pretty_move: str) -> bool:
        try:
            move = self.pretty_to_move(pretty_move)
        except PrettyMoveException:
            return False
        return self.is_legal_move(move)

    def first_wins(self) -> bool:
        return c_bool(self.__lib.firstWins(self.__pointer)).value

    def second_wins(self) -> bool:
        return c_bool(self.__lib.secondWins(self.__pointer)).value

    def first_score(self) -> int:
        return self.first_board().count('1')

    def second_score(self) -> int:
        return self.second_board().count('1')

    def flip_vertical(self) -> NoReturn:
        self.__lib.flipVertical(self.__pointer)

    def mirror_horizontal(self) -> NoReturn:
        self.__lib.mirrorHorizontal(self.__pointer)

    def flip_diagonal(self) -> NoReturn:
        self.__lib.flipDiagonal(self.__pointer)

    def rotate90(self) -> NoReturn:
        self.__lib.rotate90(self.__pointer)

    def rotate180(self) -> NoReturn:
        self.__lib.rotate180(self.__pointer)

    def rotate270(self) -> NoReturn:
        self.__lib.rotate270(self.__pointer)

    def transform(self) -> NoReturn:
        self.__lib.transform(self.__pointer)

    def reset_transformation(self) -> NoReturn:
        self.__lib.resetTransformation(self.__pointer)

    def format(self) -> str:
        first_board_str = self.first_board()
        second_board_str = self.second_board()
        first_board = int(first_board_str, 2)
        second_board = int(second_board_str, 2)
        neutral_board = int(self.neutral_board(), 2)
        legal_board = int(self.legal_board(), 2)

        layers = []
        for y in range(1, self.__size + 1):
            layer_size = y
            layer_shift = sum([i * i for i in range(1, layer_size)])
            y_char = self.__y_char(self.__size, y)
            rows = [
                ' '.join([y_char + '|'] + [self.__x_char(layer_size, x) for x in range(1, layer_size + 1)]),
                '--'.join(['' for _ in range(layer_size + 2)])
            ]
            for z in range(1, layer_size + 1):
                z_shift = (z - 1) * layer_size
                row = [self.__z_char(layer_size, z) + '|']
                for x in range(1, layer_size + 1):
                    x_shift = x - 1
                    shift = layer_shift + z_shift + x_shift
                    symbol = self.__empty_symbol
                    if (first_board >> shift) & 1 == 1:
                        symbol = self.__first_symbol
                    elif (second_board >> shift) & 1 == 1:
                        symbol = self.__second_symbol
                    elif (neutral_board >> shift) & 1 == 1:
                        symbol = self.__neutral_symbol
                    elif (legal_board >> shift) & 1 == 1:
                        symbol = self.__legal_symbol
                    row.append(symbol)
                rows.append(' '.join(row))
            layers.append(rows)
        layers = ['\n'.join(layer) for layer in layers]
        board = '\n\n'.join(layers)

        first_stats = '-> ' if not self.is_over() and self.is_first_turn() else '   '
        second_stats = '-> ' if not self.is_over() and self.is_second_turn() else '   '

        first_stats += f'{self.__first_symbol}: {self.first_score()}'
        second_stats += f'{self.__second_symbol}: {self.second_score()}'

        first_pretty_moves = list(self.first_pretty_moves())
        second_pretty_moves = list(self.second_pretty_moves())

        first_stats += f' [{first_pretty_moves[-1]}]' if len(first_pretty_moves) > 0 else ''
        second_stats += f' [{second_pretty_moves[-1]}]' if len(second_pretty_moves) > 0 else ''

        first_stats += ' Win!' if self.first_wins() else ''
        second_stats += ' Win!' if self.second_wins() else ''

        board += f'\n\n{first_stats}\n{second_stats}'
        return board

    def move_to_pretty(self, move: int) -> str:
        if move not in self.__moves_pretty.keys():
            raise PrettyMoveException()
        return self.__moves_pretty[move]

    def pretty_to_move(self, pretty_move: str) -> int:
        if pretty_move not in self.__pretty_moves.keys():
            raise PrettyMoveException()
        return self.__pretty_moves[pretty_move]

    def __del__(self):
        self.__lib.destroy(self.__pointer)
