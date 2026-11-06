"""Controlador difús amb paràmetres optimitzats per un algorisme genètic.

Executa amb --entrenar per generar els paràmetres; sense arguments, juga
amb els darrers paràmetres desats.
"""

import json
import random

import arcade
from esquivar_fuzzy import FuzzyGame
from joc_comu import OBSERVATION_VERSION, ROOT, SCALE
from panell_entrenament import TrainingPanel
from entrenament_comu import evaluate, evaluate_population, evolve
from genetica_comuna import FUZZY_BOUNDS, make_population, make_toolbox
from opcions_entrenament import parser_entrenament, valida_opcions

PARAMETERS_FILE = ROOT / "fuzzy_genetic.json"
BOUNDS = FUZZY_BOUNDS


class AdaptiveFuzzyGame(FuzzyGame):
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.parameters = parameters
        self.training_visual = False
        self.training_cancelled = False
        self.training_status = ""
        self.training_history = []
        self.training_snapshot = None
        self.training_panel = None
        self.training_panel_visible = True
        self.inspected_player = None
        self._inspect_controller = False

    def on_key_press(self, key, modifiers):
        if self.training_visual:
            if key == arcade.key.ESCAPE:
                self.training_cancelled = True
            elif key == arcade.key.P:
                self.training_panel_visible = not self.training_panel_visible
            return
        super().on_key_press(key, modifiers)

    def on_close(self):
        if self.training_visual:
            self.training_cancelled = True
        else:
            super().on_close()

    def on_draw(self):
        if not self.training_visual:
            return super().on_draw()
        self.clear()
        self.visible_players.draw()
        self.projectiles.draw()
        if self.inspected_player and self.inspected_player.alive:
            player = self.inspected_player
            arcade.draw_circle_outline(player.center_x, player.center_y,
                                       max(player.width, player.height) * 0.65,
                                       arcade.color.YELLOW, 3 * SCALE)
        if not self.training_panel_visible:
            return
        if self.training_panel is None:
            self.training_panel = TrainingPanel()
        self.training_panel.draw(self)

    def fuzzy_parameters(self):
        return self.parameters[:2]

    def player_direction(self, index):
        if index == 0:
            direction = self.fuzzy_direction()
            if abs(direction) <= self.parameters[2]:
                return 0
            return 1 if direction > 0 else -1
        return super().player_direction(index)




def train(population_size, generations, max_frames, seed, visual=False, visual_speed=4, trials=1):
    random.seed(seed)
    toolbox = make_toolbox('fuzzy', prefix='Fuzzy')
    game = AdaptiveFuzzyGame([400, 70, 0.15])
    game.training_visual = visual
    game.visual_speed = visual_speed
    game.training_without_graphics = not visual
    game.set_visible(visual)
    try:
        population = make_population(toolbox, population_size)
        population[0][:] = [400, 70, 0.15]
        best, seeds, validation_seeds, validation = evolve(
            game, toolbox, population, seed, trials, generations, max_frames, visual
        )
        result = {"observation_version": OBSERVATION_VERSION,
                  "validation_seeds": validation_seeds, "validation_mean_frames": validation,
                  "reaction_distance": best[0], "safety_margin": best[1],
                  "movement_threshold": best[2],
                  "mean_survival_frames": best.fitness.values[0],
                  "seeds": seeds, "max_frames": max_frames}
        PARAMETERS_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Paràmetres desats a {PARAMETERS_FILE}: {result}")
    finally:
        game.close()


def main():
    parser = parser_entrenament(__doc__)
    args = parser.parse_args()
    valida_opcions(parser, args)
    if args.entrenar:
        try:
            train(args.poblacio, args.generacions, args.fotogrames,
                  args.llavor if args.llavor is not None else 42, visual=args.visual,
                  visual_speed=args.velocitat, trials=args.proves)
        except KeyboardInterrupt:
            print("Entrenament interromput; no s'han desat paràmetres nous.")
        return
    if not PARAMETERS_FILE.exists():
        parser.error("Primer entrena el controlador amb --entrenar")
    data = json.loads(PARAMETERS_FILE.read_text(encoding="utf-8"))
    if data.get("observation_version") != OBSERVATION_VERSION:
        parser.error("El model usa observacions antigues; torna a entrenar-lo amb --entrenar")
    game = AdaptiveFuzzyGame([data["reaction_distance"], data["safety_margin"],
                              data["movement_threshold"]], seed=args.llavor)
    game.setup()
    game.run()


if __name__ == "__main__":
    main()
