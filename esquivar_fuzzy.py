"""Joc amb el primer jugador controlat per un controlador de lògica difusa."""

import arcade
import numpy as np
import skfuzzy as fuzz

from esquivar import Game
from joc_comu import BASE_HEIGHT, observe, parse_seed

UNIVERSE = np.linspace(0, 1, 101)
CLOSE = fuzz.trimf(UNIVERSE, [0, 0, 1])
ALIGNED = fuzz.trimf(UNIVERSE, [0, 0, 1])


class FuzzyGame(Game):
    def player_direction(self, index):
        if index == 0:
            direction = self.fuzzy_direction()
            return 0 if abs(direction) <= 0.15 else (1 if direction > 0 else -1)
        return super().player_direction(index)

    def fuzzy_parameters(self):
        return 400, 70

    def fuzzy_direction(self):
        """Combina la proximitat i el risc lateral de cada meteorit.

        Els graus de pertinença varien entre zero i un. Els meteorits propers
        i alineats amb la nau exerceixen més influència sobre la direcció.
        """
        player = self.players[0]
        position, *threats = observe(player, self.projectiles)
        reaction_distance, safety_margin = self.fuzzy_parameters()
        # Els paràmetres són posicions relatives al camp, sense velocitat.
        proximity_limit = reaction_distance / BASE_HEIGHT
        lateral_limit = (60 + safety_margin) / 120
        push_left = 0.0
        push_right = 0.0
        details = []
        for offset, height_fraction in zip(threats[::2], threats[1::2]):
            distance = min(1.0, height_fraction / proximity_limit)
            near = fuzz.interp_membership(UNIVERSE, CLOSE, distance)
            lateral_distance = min(1.0, abs(offset) / lateral_limit)
            aligned = fuzz.interp_membership(UNIVERSE, ALIGNED, lateral_distance)
            risk = min(near, aligned)
            if getattr(self, "_inspect_controller", False):
                details.append({"height": height_fraction, "offset": abs(offset),
                                "near": float(near), "aligned": float(aligned),
                                "risk": float(risk)})
            if offset >= 0:
                push_left = max(push_left, risk)
            else:
                push_right = max(push_right, risk)

        # Les vores també es dedueixen de la posició observada.
        if push_left and position < -0.9:
            push_right = max(push_right, push_left)
            push_left = 0
        if push_right and position > 0.9:
            push_left = max(push_left, push_right)
            push_right = 0
        result = push_right - push_left
        if getattr(self, "_inspect_controller", False):
            self.controller_debug = {"kind": "fuzzy", "observations": (position, *threats),
                                     "parameters": list(self.parameters), "left": push_left,
                                     "right": push_right, "result": result,
                                     "proximity_limit": proximity_limit,
                                     "lateral_limit": lateral_limit,
                                     "threat": max(
                                         (item for item in details if item["height"] < 1),
                                         key=lambda item: item["risk"], default=None)}
        return result

    def on_key_press(self, key, modifiers):
        # El primer jugador està reservat al controlador difús.
        if self.game_over or key not in (arcade.key.A, arcade.key.S):
            super().on_key_press(key, modifiers)


if __name__ == "__main__":
    game = FuzzyGame(seed=parse_seed())
    game.setup()
    game.run()
