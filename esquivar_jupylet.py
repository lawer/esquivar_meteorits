import os
import random
import sys

from jupylet.label import Label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupylet.state import State
from jupylet.app import App
from jupylet.sprite import Sprite
import jupylet.rl

esquivar = jupylet.rl.GameProcess('esquivar_jupylet')

app = App()

ufo = Sprite("images/ufo_green.png", x=app.width / 2, y=80)
projectiles = []
spawn_points = [
    ((app.width * i) // 8, app.height)
    for i in range(1, 9)
]

state = State(
    left=False,
    right=False,
    vx=0,
    score=0,
    lives=3
)

START = app.save_state('esquivar_jupylet', 'esquivar_jupylet.state', state)

lives = Label(
    f'Score: {state.lives}', font_size=24, color="red",
    x=10, y=app.height - 32, anchor_x='left'
)

score = Label(
    f'Score: {state.score}', font_size=24,
    x=app.width - 10, y=app.height - 32, anchor_x='right'
)


@app.event
def key_event(key, action, modifiers):
    keys = app.window.keys

    if action == keys.ACTION_PRESS:

        if key == keys.LEFT:
            state.left = True

        if key == keys.RIGHT:
            state.right = True

    if action == keys.ACTION_RELEASE:

        if key == keys.LEFT:
            state.left = False

        if key == keys.RIGHT:
            state.right = False


@app.run_me_every(1 / 120)
def update_ufo(ct, dt):
    if state.left:
        state.vx = -300

    elif state.right:
        state.vx = 300
    else:
        state.vx = 0

    ufo.x = ufo.x + state.vx * dt
    ufo.clip_position(app.width, app.height)


@app.run_me_every(1 / 2)
def create_projectile(ct, dt):
    spawn_point = random.choice(spawn_points)

    projectile = Sprite("images/meteor.png", x=spawn_point[0], y=spawn_point[1])
    projectiles.append(projectile)


@app.run_me_every(1 / 60)
def update_projectiles(ct, dt):
    for projectile in projectiles:
        projectile.y -= 300 * dt

        if projectile.y < 0:
            projectiles.remove(projectile)
            state.score += 1
            score.text = f'Score: {state.score}'

        if len(projectile.collisions_with(ufo)) > 0:
            projectiles.remove(projectile)
            state.lives -= 1
            lives.text = f'Lives: {state.lives}'

            if state.lives <= 0:
                app.stop()


@app.event
def render(ct, dt):
    app.window.clear()

    ufo.draw()

    for projectile in projectiles:
        projectile.draw()

    lives.draw()
    score.draw()


def step(player=[0], n=1):
    print('step', player, n)
    if player[0] < -0.1:
        state.left = True
        state.right = False
    elif player[0] > 0.1:
        state.left = False
        state.right = True
    else:
        state.left = False
        state.right = False

    if app.mode == 'hidden':
        app.step(n)

    reward = state.score

    return observe(reward)


def observe(reward=0):
    return {
        'screen0': app.observe(),
        'player': {'score': state.score, 'reward': reward},
    }


def reset():
    load(START)
    return observe()


def load(path):
    app.load_state(path, state)
    return observe()


if __name__ == '__main__':
    app.run()
