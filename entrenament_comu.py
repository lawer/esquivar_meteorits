"""Avaluació i evolució compartides pels controladors genètics."""

import time

import arcade
import copy

from esquivar import UFO
from genetica_comuna import next_population, seeds_for_trials
from joc_comu import PLAYER_SPEED, ROOT, SCALE


def run_episode(game, seed, max_frames, visual=False, status='', trial=1, trials=1):
    """Executa una partida sense modificar el rècord del joc manual."""
    game.seed = seed
    game.setup()
    game.high_score = float('inf')
    for frame in range(max_frames):
        started = time.monotonic() if visual else 0
        if visual:
            game.dispatch_events()
            if game.training_cancelled:
                raise KeyboardInterrupt
        game.on_update(1 / 60)
        if visual:
            game.training_status = (f"{status} | Prova {trial}/{trials} "
                                    f"| Fotograma {frame + 1}/{max_frames}")
            game.on_draw()
            game.flip()
            delay = 1 / 60 - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
        if not game.players[0].alive:
            return frame + 1
    return max_frames


def evaluate(game, parameters, seeds, max_frames, visual=False, status=''):
    """Mitjana de fotogrames supervivents amb les mateixes llavors."""
    game.parameters = parameters
    scores = [run_episode(game, seed, max_frames, visual=visual, status=status,
                          trial=i, trials=len(seeds))
              for i, seed in enumerate(seeds, 1)]
    return (sum(scores) / len(scores),)


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



def evolve(game, toolbox, population, seed, trials, generations, max_frames, visual, before_generation=None):
    seeds = seeds_for_trials(seed, trials)
    best = None
    print("gen  avaluacions  mitjana  millor", flush=True)
    for generation in range(generations + 1):
        if generation:
            population[:] = next_population(population, toolbox)
        if before_generation is not None:
            before_generation()
        invalid = [individual for individual in population if not individual.fitness.valid]
        if visual:
            results = evaluate_population(game, population, seeds, max_frames,
                                          generation, generations)
            for individual, fitness in zip(population, results):
                individual.fitness.values = fitness
            evaluations = len(population)
        else:
            for individual in invalid:
                individual.fitness.values = evaluate(game, individual, seeds, max_frames)
            evaluations = len(invalid)
        candidate = max(population, key=lambda item: item.fitness.values[0])
        if best is None or candidate.fitness.values[0] > best.fitness.values[0]:
            best = copy.deepcopy(candidate)
        scores = [individual.fitness.values[0] for individual in population]
        mean, maximum = sum(scores) / len(scores), max(scores)
        game.training_history.append((mean, maximum))
        print(f"{generation:3}  {evaluations:11}  {mean:7.1f}  {maximum:6.1f}", flush=True)
    validation_seeds = seeds_for_trials(seed, trials, validation=True)
    validation = evaluate(game, best, validation_seeds, max_frames)[0]
    print(f"Validació amb llavors noves: {validation:.1f} fotogrames de mitjana", flush=True)
    return best, seeds, validation_seeds, validation
