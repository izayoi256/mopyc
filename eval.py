import argparse
import gym
import logging
import numpy
import os
import random
import sys
# noinspection PyUnresolvedReferences
import ai.gym
from ai.gym.agents import create_dqn_agent
from ai.gym.envs.mosaic import MosaicObservation


class AgentPolicy:
    def __init__(self, agent, epsilon):
        self.__agent = agent
        self.epsilon = epsilon

    def __call__(self, observation: MosaicObservation):
        legal_actions = observation.legal_actions
        if numpy.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        action = self.__agent.act(observation.array)
        return action if (action in legal_actions) else random.choice(legal_actions)


def evaluate(env, agent, episodes):
    wins = 0
    first_wins = 0
    illegal_actions = 0
    for episode in range(1, episodes + 1):
        is_first_player = (episode % 2 == 1)
        observation: MosaicObservation = env.reset(
            is_first_player=is_first_player,
        )
        while True:
            action = agent.act(observation.array)
            observation, reward, done, info = env.step(action)
            if info['illegal_action']:
                illegal_actions += 1
            if done:
                if info['win']:
                    wins += 1
                    if is_first_player:
                        first_wins += 1
                break
        agent.stop_episode()
    print(
        'episode:', episodes,
        '/ illegal actions:', illegal_actions,
        '/ wins:', wins,
        '/ loses:', episodes - wins - illegal_actions,
        '/ first wins:', first_wins,
        '/ second wins:', wins - first_wins,
    )


def main(size, episodes):
    env = gym.make('Mosaic-v0', size=size)
    model_directory = os.path.join('models', 'dqn', str(size))
    agent = create_dqn_agent(env, start_epsilon=0.0, end_epsilon=0.0)
    agent.load(model_directory)
    evaluate(env, agent, episodes)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'size',
        help='Size of board.',
        type=int,
        choices=[i for i in range(1, 8)],
    )
    parser.add_argument(
        'n',
        help='Number of games.',
        type=int
    )
    args = parser.parse_args()
    main(args.size, args.n)
