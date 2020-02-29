import argparse
from chainerrl.agents import DQN
import gym
import logging
import os
import sys
# noinspection PyUnresolvedReferences
import ai.gym
from ai.gym.agents import create_dqn_agent
from ai.gym.envs.mosaic import Action, MosaicEnv, MosaicObservation, Policy, cast_action


class AgentPolicy(Policy):
    __agent: DQN
    __last_reward: float

    def __init__(self, agent: DQN):
        self.__agent = agent
        self.__last_reward = 0.0

    def __call__(self, observation: MosaicObservation) -> Action:
        while True:
            action = self.__agent.act_and_train(observation.array, self.__last_reward)
            action = cast_action(action)
            if observation.game.is_legal_move(action):
                self.__last_reward = 0.0
                return action
            self.__last_reward = -1.0


class PlayerPolicy(Policy):
    __env: MosaicEnv
    __numeric_mode: bool
    __suggest_mode: bool
    __command_width = 7

    def __init__(self, env: MosaicEnv):
        self.__env = env
        self.__numeric_mode = False

    @classmethod
    def __print_command(cls, command: str, description: str):
        print(f'[{command}]'.rjust(cls.__command_width) + f' {description}')

    def __call__(self, observation: MosaicObservation) -> Action:
        observation.game.reset_transformation()
        numeric_mode_command = 'n'
        first_moves_command = 'f'
        second_moves_command = 's'
        undo_command = 'u'
        exit_command = 'x'
        while True:
            pretty_legal_moves = dict([(str(k), v) for k, v in enumerate(observation.game.pretty_legal_moves())])
            undoable = observation.game.moves_made() >= 2
            self.__env.render()
            print('\nCommands:')
            self.__print_command('A1a', 'Make move')
            if self.__numeric_mode:
                for k, v in pretty_legal_moves.items():
                    self.__print_command(k, v)

            self.__print_command(numeric_mode_command, f'{"Disable" if self.__numeric_mode else "Enable"} Numeric mode')
            self.__print_command(first_moves_command, 'Show O moves')
            self.__print_command(second_moves_command, 'Show X moves')
            self.__print_command(numeric_mode_command, f'{"Disable" if self.__numeric_mode else "Enable"} Numeric mode')
            if undoable:
                self.__print_command(undo_command, 'Undo')
            self.__print_command(exit_command, 'Exit')
            while True:
                print('\n\u001B[1F', end='')
                command = input('> ').strip()
                print('\u001B[K', end='')
                if command == '':
                    print('\u001B[1F', end='')
                    print('\u001B[K', end='')
                    continue
                if command == numeric_mode_command:
                    self.__numeric_mode = not self.__numeric_mode
                    break
                if command == first_moves_command:
                    print(observation.game.first_pretty_moves(), end='')
                    print('\u001B[1F', end='')
                    print('\u001B[K', end='')
                    continue
                if command == second_moves_command:
                    print(observation.game.second_pretty_moves(), end='')
                    print('\u001B[1F', end='')
                    print('\u001B[K', end='')
                    continue
                if undoable and command == undo_command:
                    observation.game.undo(2)
                    break
                if command == exit_command:
                    sys.exit(0)
                if self.__numeric_mode and command in pretty_legal_moves.keys():
                    pretty_move = pretty_legal_moves[command]
                else:
                    pretty_move = command
                if not observation.game.is_legal_pretty_move(pretty_move):
                    print('Illegal command. Try again.', end='')
                    print('\u001B[1F', end='')
                    print('\u001B[K', end='')
                    continue
                return observation.game.pretty_to_move(pretty_move)


def play(env, p1_policy: Policy, p2_policy: Policy):
    observation: MosaicObservation = env.reset(
        opponent_policy=p2_policy,
    )
    while True:
        action = p1_policy(observation)
        observation, _, done, _ = env.step(action)
        if done:
            break
    env.render()


def main(size: int, p1: bool, p2: bool):
    env = gym.make('Mosaic-v0', size=size, done_on_illegal_move=False)

    def create_agent():
        try:
            model_directory = os.path.join('models', 'dqn', str(size))
            agent = create_dqn_agent(env, start_epsilon=0.0, end_epsilon=0.0)
            agent.load(model_directory)
            return agent
        except Exception:
            return None

    p1_policy = PlayerPolicy(env) if p1 else AgentPolicy(create_agent())
    p2_policy = PlayerPolicy(env) if p2 else AgentPolicy(create_agent())

    play(env, p1_policy, p2_policy)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'size',
        help='Size of board. (Default=7)',
        type=int,
        choices=[i for i in range(3, 8)],
        default=7,
    )
    parser.add_argument(
        '-p1',
        help='Player 1',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-p2',
        help='Player 2',
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    main(args.size, args.p1, args.p2)
