"""Entrenament sincronitzat de fuzzy, fuzzy genètic, xarxa genètica i NEAT."""

import argparse
import copy
import json
import math
import queue
import random
import threading
from pathlib import Path

import arcade
import neat
from deap import algorithms, base, creator, tools

from esquivar import UFO
from esquivar_fuzzy import FuzzyGame
from esquivar_xarxa_genetica import NetworkWorker, NeuralGame
from joc_comu import HEIGHT, PLAYER_SPEED, PROJECTILE_FREQUENCY, ROOT, SCALE, WIDTH, create_projectile, observe


class FuzzyAgent:
    fuzzy_direction = FuzzyGame.fuzzy_direction

    def __init__(self, sprite, parameters, projectiles):
        self.players = [sprite]
        self.parameters = parameters
        self.projectiles = projectiles

    def fuzzy_parameters(self):
        return self.parameters[:2]

    def direction(self):
        result = self.fuzzy_direction()
        return 0 if abs(result) <= self.parameters[2] else (1 if result > 0 else -1)


class NeuralAgent:
    player_direction = NeuralGame.player_direction

    def __init__(self, sprite, parameters, projectiles, worker):
        self.players = [sprite]
        self.parameters = parameters
        self.projectiles = projectiles
        self.worker = worker

    def direction(self):
        return self.player_direction(0)


def make_toolbox(kind, count=0):
    if not hasattr(creator, 'ComparisonFitness'):
        creator.create('ComparisonFitness', base.Fitness, weights=(1.0,))
        creator.create('ComparisonIndividual', list, fitness=creator.ComparisonFitness)
    toolbox = base.Toolbox()
    if kind == 'fuzzy':
        bounds = [(150., 800.), (20., 220.), (0.01, 0.45)]
        toolbox.register('individual', tools.initIterate, creator.ComparisonIndividual,
                         lambda: [random.uniform(a, b) for a, b in bounds])
        toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=[100, 30, 0.06], indpb=0.4)
    else:
        bounds = [(-5., 5.)] * count
        toolbox.register('individual', tools.initIterate, creator.ComparisonIndividual,
                         lambda: [random.uniform(-1, 1) for _ in range(count)])
        toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.15, indpb=0.15)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxBlend, alpha=0.3)
    toolbox.register('select', tools.selTournament, tournsize=3)

    def bounded(operator):
        def wrapped(*args, **kwargs):
            children = operator(*args, **kwargs)
            for child in children:
                child[:] = [max(lo, min(hi, value)) for value, (lo, hi) in zip(child, bounds)]
            return children
        return wrapped
    toolbox.decorate('mate', bounded)
    toolbox.decorate('mutate', bounded)
    return toolbox


class ComparisonWindow(arcade.Window):
    def __init__(self, args, worker):
        super().__init__(WIDTH, HEIGHT, 'Comparació sincronitzada',
                         update_rate=1 / (60 * args.velocitat), antialiasing=False)
        self.args, self.worker = args, worker
        self.training_seed = args.llavor
        random.seed(args.llavor)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, str(ROOT / 'config-file.txt'))
        config.pop_size = args.poblacio
        config.no_fitness_termination = True
        self.neat_config = config
        self.neat = neat.Population(config)
        count = worker.request('count')
        self.toolbox_fuzzy = make_toolbox('fuzzy')
        self.toolbox_neural = make_toolbox('neural', count)
        fuzzy = self.toolbox_fuzzy.population(n=args.poblacio)
        fuzzy[0][:] = [400, 70, 0.15]
        neural = self.toolbox_neural.population(n=args.poblacio)
        self.groups = [
            {'name': 'FUZZY FIX (referencia)', 'kind': 'baseline', 'population': [[400, 70, 0.15]], 'best': None},
            {'name': 'FUZZY GENETIC', 'kind': 'fuzzy', 'population': fuzzy, 'best': None},
            {'name': 'XARXA GENETICA', 'kind': 'neural', 'population': neural, 'best': None},
            {'name': 'NEAT', 'kind': 'neat', 'population': [], 'best': None},
        ]
        self.generation = 0
        self.trial = 0
        self.frame = 0
        self.score = 0
        self.completed = False
        self._shutdown = False
        self.cancelled = threading.Event()
        self.generation_ready = queue.Queue()
        self.generation_done = threading.Event()
        self.state = 'waiting'
        self.summary = []
        self.best_ever = None
        self.panel_text = {}
        self.projectiles = arcade.SpriteList()
        self.rng = random.Random(args.llavor)
        self.window_camera = arcade.Camera2D(window=self)
        header = 80 * SCALE
        half_w = WIDTH / 2
        half_h = (HEIGHT - header) / 2
        self.cameras = []
        for i in range(4):
            col, row = i % 2, 1 - i // 2
            viewport = arcade.LBWH(col * half_w, row * half_h, half_w, half_h)
            self.cameras.append(arcade.Camera2D(
                viewport=viewport, position=(WIDTH / 2, HEIGHT / 2),
                projection=arcade.LRBT(-WIDTH / 2, WIDTH / 2, -HEIGHT / 2, HEIGHT / 2),
                window=self))
        self.thread = threading.Thread(target=self.evolve_neat, daemon=True)
        self.thread.start()

    def evolve_neat(self):
        def evaluate(genomes, config):
            if self.cancelled.is_set():
                raise InterruptedError
            self.generation_done.clear()
            self.generation_ready.put(('generation', genomes))
            self.generation_done.wait()
            if self.cancelled.is_set():
                raise InterruptedError
        try:
            self.neat.run(evaluate, self.args.generacions + 1)
            self.generation_ready.put(('finished', None))
        except InterruptedError:
            pass
        except Exception as exc:
            self.generation_ready.put(('error', exc))

    def new_generation(self, genomes):
        self.groups[3]['population'] = [genome for _, genome in genomes]
        self.worker.reset()
        for group in self.groups:
            group['totals'] = [0] * len(group['population'])
        self.trial = 0
        self.new_trial()

    def new_trial(self):
        self.trial += 1
        self.frame = 0
        self.score = 0
        self.counter = 0
        self.rng = random.Random(self.training_seed + self.trial - 1)
        self.projectiles = arcade.SpriteList()
        for group in self.groups:
            group['sprites'] = arcade.SpriteList()
            group['agents'] = []
            group['round'] = [None] * len(group['population'])
            for candidate in group['population']:
                sprite = UFO(str(ROOT / 'images/ufoGreen.png'), SCALE, arcade.color.GREEN)
                group['sprites'].append(sprite)
                if group['kind'] in ('baseline', 'fuzzy'):
                    controller = FuzzyAgent(sprite, candidate, self.projectiles)
                elif group['kind'] == 'neural':
                    controller = NeuralAgent(sprite, candidate, self.projectiles, self.worker)
                else:
                    controller = neat.nn.FeedForwardNetwork.create(candidate, self.neat_config)
                group['agents'].append((sprite, controller))
        self.state = 'running'

    def on_update(self, delta_time):
        if self.state == 'waiting':
            try:
                kind, payload = self.generation_ready.get_nowait()
            except queue.Empty:
                return
            if kind == 'generation':
                self.new_generation(payload)
            elif kind == 'error':
                raise RuntimeError('Error durant l’evolució NEAT') from payload
            else:
                self.state = 'finished'
                self.completed = True
                self.save_results()
            return
        if self.state != 'running':
            return
        self.frame += 1
        self.counter += 1
        if self.counter >= PROJECTILE_FREQUENCY:
            self.projectiles.append(create_projectile(self.rng, self.score))
            self.counter = 0
        self.projectiles.update()
        for meteor in list(self.projectiles):
            if meteor.top < 0:
                meteor.kill()
                self.score += 1
        alive = 0
        for group in self.groups:
            for i, (sprite, controller) in enumerate(group['agents']):
                if group['round'][i] is not None:
                    continue
                if group['kind'] == 'neat':
                    result = controller.activate(observe(sprite, self.projectiles))[0]
                    direction = 1 if result > 0.5 else -1 if result < -0.5 else 0
                else:
                    direction = controller.direction()
                sprite.change_x = direction * PLAYER_SPEED
                sprite.update()
                if arcade.check_for_collision_with_list(sprite, self.projectiles, method=3):
                    group['round'][i] = self.frame
                    sprite.kill()
                else:
                    alive += 1
        if not alive or self.frame >= self.args.fotogrames:
            self.finish_trial()

    def finish_trial(self):
        for group in self.groups:
            for i, result in enumerate(group['round']):
                group['totals'][i] += self.frame if result is None else result
        if self.trial < self.args.proves:
            self.new_trial()
            return
        generation_results = {}
        for group in self.groups:
            scores = [total / self.args.proves for total in group['totals']]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            best_score = scores[best_index]
            if group['kind'] == 'neat':
                for candidate, score in zip(group['population'], scores):
                    candidate.fitness = score
            elif group['kind'] != 'baseline':
                for candidate, score in zip(group['population'], scores):
                    candidate.fitness.values = (score,)
            if group['best'] is None or best_score > group['best']:
                group['best'] = best_score
                group['best_candidate'] = copy.deepcopy(group['population'][best_index])
            generation_results[group['name']] = {'best': best_score, 'mean': sum(scores) / len(scores)}
            if self.best_ever is None or best_score > self.best_ever['fitness']:
                self.best_ever = {'controllers': [group['name']], 'fitness': best_score,
                                  'generation': self.generation}
            elif best_score == self.best_ever['fitness'] and group['name'] not in self.best_ever['controllers']:
                self.best_ever['controllers'].append(group['name'])
        self.summary.append({'generation': self.generation, 'results': generation_results})
        print(f"Generació {self.generation}: " + ', '.join(
            f"{name} {stats['best']:.1f}" for name, stats in generation_results.items()), flush=True)
        if self.generation < self.args.generacions:
            for group, toolbox in ((self.groups[1], self.toolbox_fuzzy),
                                   (self.groups[2], self.toolbox_neural)):
                population = group['population']
                elites = [toolbox.clone(ind) for ind in tools.selBest(population, 2)]
                selected = toolbox.select(population, len(population) - len(elites))
                group['population'] = elites + algorithms.varAnd(selected, toolbox, cxpb=0.5, mutpb=0.7)
            self.generation += 1
        self.state = 'waiting'
        self.generation_done.set()

    def live_leaders(self):
        """Millor mitjana provisional de les proves iniciades, sense barrejar individus."""
        if not self.trial or any('round' not in group for group in self.groups):
            return [], 0, []
        values = []
        for group in self.groups:
            best = max((total + (self.frame if current is None else current)) / self.trial
                       for total, current in zip(group['totals'], group['round']))
            values.append(best)
        maximum = max(values)
        return [i for i, value in enumerate(values) if value == maximum], maximum, values

    def label(self, key, text, x, y, size=12, color=arcade.color.WHITE):
        if key not in self.panel_text:
            self.panel_text[key] = arcade.Text('', x, y, color, size * SCALE)
        item = self.panel_text[key]
        if item.text != text:
            item.text = text
        item.draw()

    def on_draw(self):
        self.clear()
        header = 80 * SCALE
        half_w, half_h = WIDTH / 2, (HEIGHT - header) / 2
        leaders, provisional, live_scores = self.live_leaders() if self.state == 'running' else ([], 0, [])
        for i, (group, camera) in enumerate(zip(self.groups, self.cameras)):
            camera.use()
            self.projectiles.draw()
            if 'sprites' in group:
                group['sprites'].draw()
            self.window_camera.use()
            col, row = i % 2, 1 - i // 2
            x, y = col * half_w, row * half_h
            arcade.draw_lbwh_rectangle_filled(x, y + half_h - 32*SCALE, half_w, 32*SCALE, (15, 24, 39, 230))
            if i in leaders:
                arcade.draw_lbwh_rectangle_filled(x, y + half_h - 3*SCALE, half_w, 3*SCALE, arcade.color.YELLOW)
            living = sum(result is None for result in group.get('round', []))
            last = '-' if group['best'] is None else f"{group['best']:.1f}"
            current = '-' if not live_scores else f'{live_scores[i]:.1f}'
            self.label(f'group{i}', f"{group['name']} | Vius {living} | Actual {current} | Millor {last}",
                       x + 8*SCALE, y + half_h - 23*SCALE, 10)
        self.window_camera.use()
        self.label('title', f"GENERACIO {self.generation}/{self.args.generacions}  |  PROVA {self.trial}/{self.args.proves}  |  FOTOGRAMA {self.frame}/{self.args.fotogrames}",
                   10*SCALE, HEIGHT - 25*SCALE, 13, arcade.color.YELLOW)
        if self.best_ever:
            names = self.best_ever['controllers']
            label = ' / '.join(names) if len(names) < 3 else f'empat de {len(names)} controladors'
            record = f"MILLOR ENTRENAMENT: {label}  {self.best_ever['fitness']:.1f} fot."
        else:
            record = 'MILLOR COMPLETAT: pendent de la primera generacio'
        if self.state == 'finished':
            record += '  |  FINALITZAT - ESC PER EIXIR'
        self.label('record', record, 10*SCALE, HEIGHT - 50*SCALE, 10)
        if leaders:
            names = [self.groups[i]['name'].replace(' (referencia)', '') for i in leaders]
            label = ' / '.join(names) if len(names) < 3 else f'empat de {len(names)} controladors'
            current = f"LIDER PROVISIONAL: {label}  |  {provisional:.1f} fot. de mitjana"
        else:
            current = ('ENTRENAMENT FINALITZAT' if self.completed else
                       'LIDER PROVISIONAL: pendent de la prova')
        self.label('live', current, 10*SCALE, HEIGHT - 70*SCALE, 10, arcade.color.YELLOW)

    def save_results(self):
        def genome_data(genome):
            def attrs(gene):
                return {attribute.name: getattr(gene, attribute.name)
                        for attribute in gene._gene_attributes}
            return {'key': genome.key, 'fitness': genome.fitness,
                    'nodes': [{'key': key, **attrs(node)} for key, node in genome.nodes.items()],
                    'connections': [{'key': list(key), 'innovation': connection.innovation,
                                     **attrs(connection)}
                                    for key, connection in genome.connections.items()]}
        models = {}
        for group in self.groups:
            candidate = group.get('best_candidate')
            if candidate is None:
                continue
            if group['kind'] == 'neat':
                models[group['name']] = {'observation_version': 2, **genome_data(candidate)}
            elif group['kind'] == 'neural':
                models[group['name']] = {'observation_version': 2, 'architecture': [7, 8, 3],
                                         'weights': list(candidate)}
            else:
                models[group['name']] = {'reaction_distance': candidate[0],
                                         'safety_margin': candidate[1],
                                         'movement_threshold': candidate[2]}
        data = {'seed': self.training_seed, 'trials': self.args.proves,
                'max_frames': self.args.fotogrames, 'population': self.args.poblacio,
                'generations': self.summary, 'best': self.best_ever, 'models': models}
        self.args.sortida.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Resultats desats a {self.args.sortida}', flush=True)

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        self.cancelled.set()
        self.generation_done.set()
        self.thread.join(timeout=10)
        self.worker.close()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.shutdown()
            self.close()
            arcade.exit()

    def on_close(self):
        self.shutdown()
        super().on_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrenar', action='store_true', help='la comparació sempre entrena els controladors')
    parser.add_argument('--visual', action='store_true', help='la comparació sempre és visual')
    parser.add_argument('--poblacio', type=int, default=12)
    parser.add_argument('--generacions', type=int, default=8)
    parser.add_argument('--fotogrames', type=int, default=1800)
    parser.add_argument('--proves', type=int, default=1)
    parser.add_argument('--llavor', type=int, default=42)
    parser.add_argument('--velocitat', type=float, default=4)
    parser.add_argument('--sortida', type=Path, default=ROOT / 'comparacio_visual_resultats.json')
    args = parser.parse_args()
    if args.poblacio < 3 or args.generacions < 1 or args.fotogrames < 1 or args.proves < 1:
        parser.error('Calen almenys 3 individus, 1 generació, 1 prova i 1 fotograma')
    if not math.isfinite(args.velocitat) or args.velocitat <= 0:
        parser.error('La velocitat ha de ser positiva')
    worker = NetworkWorker()
    try:
        window = ComparisonWindow(args, worker)
    except Exception:
        worker.close()
        raise
    try:
        arcade.run()
    finally:
        window.shutdown()
        if not window.closed:
            window.close()


if __name__ == '__main__':
    main()
