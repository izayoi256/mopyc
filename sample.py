import argparse
from concurrent import futures

from core import Game
from ai import RandomPlayer

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='Size of board. (Default=7)', type=int, choices=[i for i in range(1, 8)],
                    default=7)
parser.add_argument('-n', help='Number of games. (Default=1)', type=int, default=1)
parser.add_argument('-c', '--concurrency', help='Number of max workers. (Default=1)', type=int, default=1)

args = parser.parse_args()


def run(size, player):
    game = Game(size)
    while not game.is_over():
        move = player.make_move(game)
        game.move(move)
    return game.first_wins()


def main():
    size = args.size
    n = args.n
    workers = args.concurrency
    player = RandomPlayer()
    first_wins = 0
    second_wins = 0
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        fs = [executor.submit(run, size, player) for _ in range(n)]
    for f in futures.as_completed(fs):
        if f.result():
            first_wins += 1
        else:
            second_wins += 1

    print('first: %d, second: %d' % (first_wins, second_wins))


if __name__ == '__main__':
    main()
