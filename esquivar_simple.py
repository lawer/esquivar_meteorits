"""Nau controlada per regles deterministes, sense aprenentatge ni lògica difusa."""

import arcade

from esquivar import Game
from joc_comu import observe, parse_seed

# Els valors utilitzen les mateixes observacions normalitzades que la resta.
HEIGHT_LIMIT = 0.3
LATERAL_LIMIT = 0.7
EDGE_LIMIT = 0.85


def simple_direction(observations):
    """Esquiva el meteorit més pròxim verticalment segons uns llindars fixos."""
    position, *threats = observations
    for lateral, height in zip(threats[::2], threats[1::2]):
        if height <= HEIGHT_LIMIT and abs(lateral) < LATERAL_LIMIT:
            if position <= -EDGE_LIMIT:
                return 1
            if position >= EDGE_LIMIT:
                return -1
            return -1 if lateral >= 0 else 1
    return 0


class SimpleGame(Game):
    def player_direction(self, index):
        if index == 0:
            return simple_direction(observe(self.players[0], self.projectiles))
        return super().player_direction(index)

    def on_key_press(self, key, modifiers):
        if self.game_over or key not in (arcade.key.A, arcade.key.S):
            super().on_key_press(key, modifiers)


if __name__ == '__main__':
    game = SimpleGame(seed=parse_seed())
    game.setup()
    game.run()
