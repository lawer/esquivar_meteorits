"""Dimensions i meteorits compartits pels diferents modes de joc."""

import argparse
from pathlib import Path

import arcade

ROOT = Path(__file__).resolve().parent
BASE_WIDTH, BASE_HEIGHT = 1024, 768
SCREEN_WIDTH, SCREEN_HEIGHT = arcade.get_display_size()
SCALE = min(1, SCREEN_WIDTH * 0.9 / BASE_WIDTH, SCREEN_HEIGHT * 0.9 / BASE_HEIGHT)
WIDTH, HEIGHT = int(BASE_WIDTH * SCALE), int(BASE_HEIGHT * SCALE)
PLAYER_SPEED = 5 * SCALE
PROJECTILE_SPEED = 5 * SCALE
PROJECTILE_FREQUENCY = 30
SCORE_PER_SPEED_INCREASE = 10
SPEED_INCREASE = 0.5 * SCALE
SPAWN_POINTS = [(WIDTH * i // 16, HEIGHT) for i in range(1, 17)]


class Projectile(arcade.Sprite):
    def __init__(self, filename, center_x, center_y, speed):
        super().__init__(filename, SCALE)
        self.center_x, self.center_y = center_x, center_y
        self.change_y = speed

    def update(self, delta_time=1 / 60):
        self.center_y -= self.change_y


def create_projectile(rng, score=0):
    x, y = rng.choice(SPAWN_POINTS)
    image = ROOT / f"images/meteorBrown_big{rng.randint(1, 4)}.png"
    speed = PROJECTILE_SPEED + (score // SCORE_PER_SPEED_INCREASE) * SPEED_INCREASE
    return Projectile(str(image), x, y, speed)


OBSERVATION_SIZE = 7


def observe(player, projectiles):
    """Posició normalitzada i informació dels tres impactes més pròxims.

    Cada parell conté el desplaçament lateral relatiu a la zona de perill
    i el temps fins a l'altura de la nau, normalitzat a 180 fotogrames.
    Un parell (0, 1) representa una amenaça absent o encara llunyana.
    """
    threats = [m for m in projectiles
               if m.change_y > 0 and m.bottom >= player.top - m.height]
    threats.sort(key=lambda m: max(0, m.bottom - player.top) / m.change_y)
    inputs = [2 * player.center_x / WIDTH - 1]
    for meteor in threats[:3]:
        distance = meteor.center_x - player.center_x
        danger_width = max(120 * SCALE, (meteor.width + player.width) / 2)
        lateral = max(-1.0, min(1.0, distance / danger_width))
        time_to_impact = max(0, meteor.bottom - player.top) / meteor.change_y
        inputs.extend((lateral, min(1.0, time_to_impact / 180)))
    inputs.extend((0, 1) * (3 - min(3, len(threats))))
    return tuple(inputs)


def parse_seed():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llavor", type=int, default=None,
                        help="llavor per reproduir la seqüència de meteorits")
    return parser.parse_args().llavor
