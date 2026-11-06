"""Compara els controladors amb les mateixes partides i llavors."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import neat

from esquivar_fuzzy import FuzzyGame
from esquivar_fuzzy_adaptatiu import AdaptiveFuzzyGame
from esquivar_neat import NeatPlayGame, load_winner
from esquivar_xarxa_genetica import NeuralGame, NetworkWorker
from joc_comu import ROOT


def read_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Falta {path.name}; entrena primer aquest controlador")
    return json.loads(path.read_text(encoding="utf-8"))


def run_episode(game, seed, max_frames):
    game.seed = seed
    game.setup()
    # Les partides de comparació no han d'alterar el rècord de joc.
    game.high_score = float('inf')
    for frame in range(1, max_frames + 1):
        game.on_update(1 / 60)
        if not game.players[0].alive:
            return frame
    return max_frames


def compare(seeds, max_frames):
    fuzzy_path = ROOT / 'fuzzy_genetic.json'
    neural_path = ROOT / 'xarxa_genetica.json'
    neat_path = ROOT / 'neat_genetic.json'
    fuzzy = read_model(fuzzy_path)
    neural = read_model(neural_path)
    read_model(neat_path)
    if neural.get('observation_version') != 2 or neural.get('architecture') != [7, 8, 3]:
        raise ValueError('La xarxa genètica és incompatible; torna a entrenar-la')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(ROOT / 'config-file.txt'))
    genome = load_winner(config)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    worker = NetworkWorker()
    results = {}
    try:
        if len(neural['weights']) != worker.request('count'):
            raise ValueError('La xarxa genètica té un nombre de pesos incorrecte')
        controllers = [
            ('Fuzzy', lambda: FuzzyGame()),
            ('Fuzzy genètic', lambda: AdaptiveFuzzyGame([
                fuzzy['reaction_distance'], fuzzy['safety_margin'], fuzzy['movement_threshold']])),
            ('NEAT', lambda: NeatPlayGame(network)),
            ('Xarxa genètica', lambda: NeuralGame(neural['weights'], worker)),
        ]
        for name, make_game in controllers:
            game = make_game()
            game.training_without_graphics = True
            game.set_visible(False)
            try:
                outcomes = []
                for i, seed in enumerate(seeds, 1):
                    frames = run_episode(game, seed, max_frames)
                    outcomes.append(frames)
                    print(f'\r{name}: {i}/{len(seeds)} proves', end='', flush=True)
                print()
                results[name] = outcomes
            finally:
                game.close()
    finally:
        worker.close()
    return results


def summarise(results, seeds, max_frames):
    summaries = {}
    for name, outcomes in results.items():
        # Les victòries empatades es reparteixen entre els participants.
        wins = 0.0
        for i, value in enumerate(outcomes):
            best = max(other[i] for other in results.values())
            if value == best:
                wins += 1 / sum(other[i] == best for other in results.values())
        summaries[name] = {
            'mean_frames': statistics.mean(outcomes),
            'median_frames': statistics.median(outcomes),
            'stdev_frames': statistics.stdev(outcomes) if len(outcomes) > 1 else 0,
            'survived_to_limit': sum(value == max_frames for value in outcomes),
            'shared_wins': wins,
            'frames_by_seed': dict(zip(map(str, seeds), outcomes)),
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proves', type=int, default=20)
    parser.add_argument('--fotogrames', type=int, default=1800)
    parser.add_argument('--llavor', type=int, default=20000,
                        help='primera llavor; les següents són consecutives')
    parser.add_argument('--sortida', type=Path, default=ROOT / 'comparativa_controladors.json')
    args = parser.parse_args()
    if args.proves < 1 or args.fotogrames < 1:
        parser.error('Calen almenys una prova i un fotograma')
    seeds = list(range(args.llavor, args.llavor + args.proves))
    results = compare(seeds, args.fotogrames)
    summaries = summarise(results, seeds, args.fotogrames)
    print('\nControlador         Mitjana   Mediana  Desv. est.  Al límit  Victòries*')
    for name, data in sorted(summaries.items(), key=lambda item: item[1]['mean_frames'], reverse=True):
        print(f"{name:18} {data['mean_frames']:8.1f} {data['median_frames']:9.1f} "
              f"{data['stdev_frames']:11.1f} {data['survived_to_limit']:8} {data['shared_wins']:10.1f}")
    print('* Les victòries empatades es reparteixen. Puntuació en fotogrames.')
    print('Arribar al límit no indica quant de temps més hauria sobreviscut la nau.')
    fuzzy = read_model(ROOT / 'fuzzy_genetic.json')
    if 'observation_version' not in fuzzy:
        print('Nota: el model fuzzy genètic no indica la versió de les observacions; pot ser antic.')
    model_files = ['fuzzy_genetic.json', 'neat_genetic.json', 'xarxa_genetica.json']
    result = {'seeds': seeds, 'max_frames': args.fotogrames,
              'model_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in model_files},
              'fuzzy_observation_version': fuzzy.get('observation_version'),
              'results': summaries}
    args.sortida.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Resultats desats a {args.sortida}')


if __name__ == '__main__':
    main()
