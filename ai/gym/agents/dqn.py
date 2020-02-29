import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy
from typing import Optional
from ai.gym.envs import MosaicEnv


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        n_hidden_channels = obs_size // 2
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False) -> chainerrl.action_value.ActionValue:
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h += x[:, 0] - 1.0
        return chainerrl.action_value.DiscreteActionValue(h)


def create_dqn_agent(
        env: MosaicEnv,
        start_epsilon: Optional[float] = None,
        end_epsilon: Optional[float] = None,
        decay_steps: Optional[int] = None,
        gamma: float = 0.99,
        gpu: bool = False,
) -> chainerrl.agents.DQN:
    obs_size = numpy.prod(numpy.array(list(env.observation_space.shape)))
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)
    if gpu:
        q_func.to_gpu()
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    start_epsilon = (1 / (n_actions + 18.7) * 14) if start_epsilon is None else start_epsilon
    end_epsilon = (start_epsilon / 4) if end_epsilon is None else end_epsilon
    decay_steps = ((n_actions - 6) ** 2 + 1367 * n_actions - 2950) if decay_steps is None else decay_steps
    '''
    n_actions: start_epsilon / decay_episodes / decay_steps => episodes
    3: 0.428 / 17500 / 3000 => 5000
    5: 0.19 / 65000 / 4000 => 6000
    7: 0.09 / 210000 / 6000 => 8000
    '''
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=end_epsilon,
        decay_steps=decay_steps,
        random_action_func=env.sample,
    )

    # noinspection PyUnresolvedReferences
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    return chainerrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        replay_start_size=500,
        update_interval=1,
        target_update_interval=n_actions // 2,
        phi=lambda x: x.astype(numpy.float32, copy=False),
    )
