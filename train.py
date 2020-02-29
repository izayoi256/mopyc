import argparse
from chainerrl.agents import DQN
import datetime
import gym
import logging
import os
import shutil
import sys
from typing import Callable, Optional
# noinspection PyUnresolvedReferences
import ai.gym
from ai.gym.agents import create_dqn_agent
from ai.gym.envs.mosaic import MosaicEnv, MosaicObservation, Policy

LOG_FREQ = 100
CHECKPOINT_FREQ = 1000

TEMP_MODEL_DIRECTORY = os.path.join('models', 'dqn', 'tmp')


def save_temp_agent(agent, name, suffix=''):
    dirname = os.path.join(TEMP_MODEL_DIRECTORY, '{}{}'.format(name, suffix))
    agent.save(dirname)


class AgentPolicy:
    __agent: DQN
    __last_reward: float

    def __init__(self, agent: DQN):
        self.__agent = agent
        self.__last_reward = 0.0

    def __call__(self, observation: MosaicObservation):
        while True:
            action = self.__agent.act_and_train(observation.array, self.__last_reward)
            if action in observation.legal_actions:
                self.__last_reward = 0.0
                return action
            self.__last_reward = -1.0


def train(
        env: MosaicEnv,
        agent: DQN,
        max_episodes: Optional[int],
        target_r: Optional[float],
        opponent_policy_generator: Callable[[], Policy],
):
    assert not (max_episodes is None and target_r is None)
    wins = 0
    illegal_actions = 0
    step = 0
    R = 0.0
    episode = 0

    def max_episodes_reached(e: int) -> bool:
        return max_episodes is not None and e > max_episodes

    def target_r_reached(r: float) -> bool:
        return target_r is not None and r >= target_r

    while True:
        episode += 1
        if max_episodes_reached(episode):
            break
        reward = 0.0
        is_first_player = (episode % 2 == 1)
        observation: MosaicObservation = env.reset(
            is_first_player=is_first_player,
            opponent_policy=opponent_policy_generator(),
        )
        while True:
            action = agent.act_and_train(observation.array, reward)
            observation, reward, done, info = env.step(action)
            step += 1
            R += reward
            if info['illegal_action']:
                illegal_actions += 1
            if done:
                if info['win']:
                    wins += 1
                break
        agent.stop_episode_and_train(observation.array, reward, done)
        if episode % CHECKPOINT_FREQ == 0:
            save_temp_agent(agent, episode, '_checkpoint')
        if episode % LOG_FREQ == 0:
            print(
                datetime.datetime.now(),
                'episodes:', episode,
                '/ steps:', step,
                '/ illegal actions', illegal_actions,
                '/ wins:', wins,
                '/ loses:', LOG_FREQ - wins - illegal_actions,
                '/ R:', R,
                '/ statistics:', agent.get_statistics(),
            )
            if target_r_reached(R):
                break
            wins = 0
            illegal_actions = 0
            R = 0.0


def main(
        size: int,
        max_episodes: Optional[int],
        target_r: Optional[float],
        retrain: bool = False,
        trainer: bool = False,
        self_training: bool = False,
        gpu: bool = False
):
    assert not (trainer and self_training)
    env = gym.make('Mosaic-v0', size=size)
    model_directory = os.path.join('models', 'dqn', str(size))
    trainer_model_directory = model_directory if self_training else os.path.join(model_directory, 'trainer')
    agent = create_dqn_agent(env, gpu=gpu)
    if retrain:
        agent.load(model_directory)

    def opponent_policy_generator() -> Optional[Policy]:
        if trainer or self_training:
            epsilon = 35 / size / 100
            opponent_agent = create_dqn_agent(env, start_epsilon=epsilon, end_epsilon=epsilon)
            opponent_agent.load(trainer_model_directory)
            return AgentPolicy(opponent_agent)
        else:
            return None

    shutil.rmtree(TEMP_MODEL_DIRECTORY, True)
    try:
        train(
            env,
            agent,
            target_r=target_r,
            max_episodes=max_episodes,
            opponent_policy_generator=opponent_policy_generator,
        )
    except (Exception, KeyboardInterrupt):
        save_temp_agent(agent, 'except')
        raise
    agent.save(model_directory)


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
        '-n',
        help='Max episodes.',
        type=int,
        default=None,
    )
    parser.add_argument(
        '-r',
        help='Target R.',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--retrain',
        help='Retrain model.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--trainer',
        help='Train with trainer model.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--self-training',
        help='Train with self model.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--gpu',
        help='Use GPU.',
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    main(
        args.size,
        max_episodes=args.n,
        target_r=args.r,
        retrain=args.retrain,
        trainer=args.trainer,
        self_training=args.self_training,
        gpu=args.gpu,
    )
