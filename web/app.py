from chainerrl.agents import DQN
from dataclasses import dataclass
from flask import (
    Flask,
    redirect,
    render_template,
    url_for,
    session,
)
from flask_wtf import (
    CSRFProtect,
    FlaskForm,
)
import gym
import os
from typing import Dict, NoReturn, Optional
import uuid
from wtforms import (
    RadioField,
    StringField,
)
from wtforms.validators import (
    DataRequired,
)
# noinspection PyUnresolvedReferences
import ai.gym
from ai.gym.agents import create_dqn_agent
from ai.gym.envs.mosaic import cast_action, MosaicEnv, MosaicObservation
from web.view_models import GameViewModel

GAME_SESSION_ID_KEY = 'id'
HUMAN_KEY = 'human'
AGENT_KEY = 'agent'


@dataclass
class GameSession:
    env: MosaicEnv
    observation: MosaicObservation
    agent: Optional[DQN] = None


app = Flask(__name__)
app.secret_key = os.urandom(24)
CSRFProtect(app)
game_sessions: Dict[uuid.UUID, GameSession] = {}


def has_game_session() -> bool:
    return GAME_SESSION_ID_KEY in session and session[GAME_SESSION_ID_KEY] in game_sessions.keys()


def get_game_session() -> GameSession:
    return game_sessions[session[GAME_SESSION_ID_KEY]]


def new_game_session(size: int) -> (uuid.UUID, MosaicEnv):
    env: MosaicEnv = gym.make(
        'Mosaic-v0',
        size=size,
        done_on_illegal_move=False,
    )
    try:
        model_directory = os.path.join(os.path.dirname(__file__), '..', 'models', 'dqn', str(size))
        agent = create_dqn_agent(env, start_epsilon=0.0, end_epsilon=0.0)
        agent.load(model_directory)
    except Exception:
        agent = None
    game_session = GameSession(
        env=env,
        observation=env.reset(transform=False),
        agent=agent,
    )
    while True:
        game_session_id = uuid.uuid4()
        if game_session_id not in game_sessions.keys():
            break
    session[GAME_SESSION_ID_KEY] = game_session_id
    game_sessions[game_session_id] = game_session
    return game_session_id, game_session


def update_game_session() -> NoReturn:
    if not has_game_session():
        return
    game_session = get_game_session()
    game_session.observation = game_session.env.observe(reset_transformation=True)


def remove_game_session() -> NoReturn:
    del session[GAME_SESSION_ID_KEY]


class IntroForm(FlaskForm):
    size = RadioField(
        'Size',
        choices=[(str(i), str(i)) for i in range(7, 2, -1)],
        validators=[DataRequired()],
        default='7',
    )


class CommandForm(FlaskForm):
    command = StringField(
        'Command',
        validators=[DataRequired()],
    )


@app.route('/', methods=['GET'])
def intro():
    if has_game_session():
        return redirect(url_for('index'))
    form = IntroForm()
    return render_template('intro.html', form=form)


@app.route('/', methods=['POST'])
def start():
    if has_game_session():
        return redirect(url_for('index'))
    form = IntroForm()
    if not form.validate_on_submit():
        return redirect(url_for('intro'))
    new_game_session(
        size=int(form.size.data),
    )
    return redirect(url_for('index'))


@app.route('/game', methods=['GET'])
def index():
    if not has_game_session():
        return redirect(url_for('intro'))
    game_session = get_game_session()
    return render_template(
        'index.html',
        game=GameViewModel(
            observation=game_session.observation,
            agent_available=game_session.agent is not None,
        ),
        form=CommandForm(),
    )


@app.route('/game', methods=['POST'])
def command():
    if not has_game_session():
        return redirect(url_for('intro'))
    cmd = CommandForm().command.data
    game_session = get_game_session()
    game = game_session.observation.game
    agent = game_session.agent

    if cmd == 'exit':
        remove_game_session()
        return redirect(url_for('intro'))

    if cmd == 'agent' and agent is not None:
        reward = 0.0
        while True:
            action = agent.act_and_train(game_session.env.observe().array, reward)
            action = cast_action(action)
            if game.is_legal_move(action):
                break
            reward = -1.0
        game.move(action)
    elif cmd == 'undo':
        game.undo()
    elif game.is_legal_pretty_move(cmd):
        game.pretty_move(cmd)

    update_game_session()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
