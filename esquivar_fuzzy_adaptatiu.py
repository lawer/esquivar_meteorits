"""Controlador difús amb paràmetres optimitzats per un algorisme genètic.

Executa amb --entrenar per generar els paràmetres; sense arguments, juga
amb els darrers paràmetres desats.
"""

import json
import random
import time

import arcade
from esquivar import UFO
from esquivar_fuzzy import FuzzyGame
from joc_comu import HEIGHT, PLAYER_SPEED, ROOT, SCALE
from panell_entrenament import TrainingPanel
from opcions_entrenament import parser_entrenament, valida_opcions

PARAMETERS_FILE = ROOT / "fuzzy_genetic.json"
BOUNDS = ((150.0, 800.0), (20.0, 220.0), (0.01, 0.45))


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


def evaluate(game, parameters, seeds, max_frames, visual=False, status=""):
    """Mitjana de fotogrames supervivents en les mateixes partides per candidat."""
    total = 0
    for trial, seed in enumerate(seeds, start=1):
        game.seed = seed
        game.parameters = parameters
        game.setup()
        # L'entrenament no ha de modificar el rècord del joc manual.
        game.high_score = float("inf")
        for frame in range(max_frames):
            started = time.monotonic() if visual else 0
            if visual:
                game.dispatch_events()
                if game.training_cancelled:
                    raise KeyboardInterrupt
            game.on_update(1 / 60)
            if visual:
                game.training_status = (f"{status} | Prova {trial}/{len(seeds)} "
                                        f"| Fotograma {frame + 1}/{max_frames}")
                game.on_draw()
                game.flip()
                delay = 1 / 60 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
            if not game.players[0].alive:
                break
        total += frame + 1
    return (total / len(seeds),)


def evaluate_population(game, population, seeds, max_frames, generation, generations):
    """Avalua totes les naus juntes, amb els mateixos meteorits per prova."""
    totals = [0] * len(population)
    for trial, seed in enumerate(seeds, start=1):
        game.seed = seed
        game.setup()
        game.visible_players = arcade.SpriteList()
        players = []
        for _ in population:
            player = UFO(str(ROOT / "images/ufoGreen.png"), SCALE, arcade.color.GREEN)
            players.append(player)
            game.visible_players.append(player)
        for frame in range(max_frames):
            started = time.monotonic()
            game.dispatch_events()
            if game.training_cancelled:
                raise KeyboardInterrupt
            game.switch_to()
            game.projectile_counter += 1
            if game.projectile_counter == game.projectile_frequency:
                game.create_projectile()
                game.projectile_counter = 0
            game.projectiles.update()
            for meteor in list(game.projectiles):
                if meteor.top < 0:
                    meteor.kill()
                    game.score += 1
            inspected = next((i for i, player in enumerate(players) if player.alive), None)
            game.inspected_player = players[inspected] if inspected is not None else None
            for i, player in enumerate(players):
                if not player.alive:
                    continue
                game.players[0] = player
                game.parameters = population[i]
                game._inspect_controller = i == inspected
                direction = game.player_direction(0)
                if game._inspect_controller:
                    snapshot = game.controller_debug.copy()
                    snapshot.update(index=i, action={-1: 'esquerra', 0: 'quieta', 1: 'dreta'}[direction])
                    game.training_snapshot = snapshot
                player.change_x = direction * PLAYER_SPEED
                player.update()
                totals[i] += 1
                if arcade.check_for_collision_with_list(player, game.projectiles):
                    player.alive = False
                    player.kill()
            game._inspect_controller = False
            alive = sum(player.alive for player in players)
            game.training_status = (f"Gen. {generation}/{generations}  Prova {trial}/{len(seeds)}  "
                                    f"Naus {alive}/{len(players)}  Fot. {frame + 1}")
            game.switch_to()
            game.on_draw()
            game.flip()
            speed = getattr(game, "visual_speed", 1)
            delay = 1 / (60 * speed) - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            if not alive:
                break
    return [(total / len(seeds),) for total in totals]


def train(population_size, generations, max_frames, seed, visual=False, visual_speed=4, trials=1):
    # DEAP només és necessari per entrenar, no per jugar amb el resultat.
    from deap import algorithms, base, creator, tools

    random.seed(seed)
    if not hasattr(creator, "FuzzyFitness"):
        creator.create("FuzzyFitness", base.Fitness, weights=(1.0,))
        creator.create("FuzzyIndividual", list, fitness=creator.FuzzyFitness)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.FuzzyIndividual,
                     lambda: [random.uniform(low, high) for low, high in BOUNDS])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0, sigma=[100, 30, 0.06], indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def bounded(operator):
        def wrapped(*args, **kwargs):
            offspring = operator(*args, **kwargs)
            for individual in offspring:
                for i, (low, high) in enumerate(BOUNDS):
                    individual[i] = max(low, min(high, individual[i]))
            return offspring
        return wrapped

    toolbox.decorate("mate", bounded)
    toolbox.decorate("mutate", bounded)
    game = AdaptiveFuzzyGame([400, 70, 0.15])
    game.training_visual = visual
    game.visual_speed = visual_speed
    game.training_without_graphics = not visual
    game.set_visible(visual)
    try:
        seeds = [seed + i for i in range(trials)]
        population = toolbox.population(n=population_size)
        # Incloure els valors originals proporciona una referència coneguda.
        population[0][:] = [400, 70, 0.15]
        best = tools.HallOfFame(1)
        stats = tools.Statistics(lambda individual: individual.fitness.values[0])
        stats.register("mitjana", lambda values: sum(values) / len(values))
        stats.register("millor", max)
        print("gen  avaluacions  mitjana  millor")
        for generation in range(generations + 1):
            if generation:
                elites = [toolbox.clone(individual) for individual in tools.selBest(population, 2)]
                selected = toolbox.select(population, len(population) - len(elites))
                population[:] = elites + algorithms.varAnd(selected, toolbox, cxpb=0.5, mutpb=0.7)
            invalid = [individual for individual in population if not individual.fitness.valid]
            if visual:
                # Reavaluem també els individus ja coneguts per mostrar tota la generació.
                results = evaluate_population(game, population, seeds, max_frames,
                                              generation, generations)
                for individual, fitness in zip(population, results):
                    individual.fitness.values = fitness
                evaluations = len(population)
            else:
                for individual in invalid:
                    individual.fitness.values = evaluate(game, individual, seeds, max_frames)
                evaluations = len(invalid)
            best.update(population)
            record = stats.compile(population)
            game.training_history.append((record['mitjana'], record['millor']))
            print(f"{generation:3}  {evaluations:11}  "
                  f"{record['mitjana']:7.1f}  {record['millor']:6.1f}", flush=True)
        validation_seeds = [seed + 10000 + i for i in range(trials)]
        validation = evaluate(game, best[0], validation_seeds, max_frames)[0]
        print(f"Validació amb llavors noves: {validation:.1f} fotogrames de mitjana", flush=True)
        result = {"validation_seeds": validation_seeds, "validation_mean_frames": validation,
                  "reaction_distance": best[0][0], "safety_margin": best[0][1],
                  "movement_threshold": best[0][2],
                  "mean_survival_frames": best[0].fitness.values[0],
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
    game = AdaptiveFuzzyGame([data["reaction_distance"], data["safety_margin"],
                              data["movement_threshold"]], seed=args.llavor)
    game.setup()
    game.run()


if __name__ == "__main__":
    main()
