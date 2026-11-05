"""Xarxa neuronal d'AITK amb pesos evolucionats amb DEAP."""

import json
import random
import subprocess
import sys

import numpy as np
from deap import algorithms, base, creator, tools

from esquivar_fuzzy_adaptatiu import AdaptiveFuzzyGame, evaluate, evaluate_population
from joc_comu import ROOT, observe
from opcions_entrenament import parser_entrenament, valida_opcions

MODEL_FILE = ROOT / "xarxa_genetica.json"
INPUTS, HIDDEN, OUTPUTS = 7, 8, 3


class NetworkWorker:
    """Manté TensorFlow fora del procés que gestiona OpenGL."""
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "xarxa_aitk_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self.models = {}
        self.spec = None

    def request(self, command, **kwargs):
        try:
            self.process.stdin.write(json.dumps({"command": command, **kwargs}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError as exc:
            raise RuntimeError("El procés de la xarxa AITK s'ha tancat inesperadament") from exc
        if not line:
            raise RuntimeError("El procés de la xarxa AITK s'ha tancat inesperadament")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"Error de la xarxa AITK: {response['error']}")
        return response["result"]

    def predict(self, weights, inputs):
        # AITK defineix les capes i l'ordre dels pesos; NumPy avalua la mateixa
        # xarxa localment per evitar una comunicació per nau i fotograma.
        key = tuple(weights)
        if key not in self.models:
            if self.spec is None:
                self.spec = self.request("spec")
                expected = [[INPUTS, HIDDEN], [HIDDEN], [HIDDEN, OUTPUTS], [OUTPUTS]]
                if (self.spec["shapes"] != expected or
                        self.spec["activations"] != ["sigmoid", "linear"]):
                    raise RuntimeError("L'arquitectura AITK no és compatible amb l'avaluador")
            data = np.asarray(weights, dtype=np.float32)
            sizes = [int(np.prod(shape)) for shape in self.spec["shapes"]]
            if len(data) != sum(sizes):
                raise ValueError("El nombre de pesos de la xarxa no és correcte")
            offsets = np.cumsum([0] + sizes)
            self.models[key] = tuple(
                data[offsets[i]:offsets[i + 1]].reshape(shape)
                for i, shape in enumerate(self.spec["shapes"])
            )
        w1, b1, w2, b2 = self.models[key]
        hidden = np.asarray(inputs, dtype=np.float32) @ w1 + b1
        hidden = 1.0 / (1.0 + np.exp(-hidden))
        return hidden @ w2 + b2

    def reset(self):
        self.models.clear()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        self.process.stdout.close()


class NeuralGame(AdaptiveFuzzyGame):
    def __init__(self, parameters, worker, seed=None, observation_version=2):
        super().__init__(parameters, seed=seed)
        self.worker = worker
        self.observation_version = observation_version

    def player_direction(self, index):
        if index != 0:
            return super().player_direction(index)
        player = self.players[0]
        inputs = observe(player, self.projectiles)
        outputs = self.worker.predict(self.parameters, inputs)
        if getattr(self, "_inspect_controller", False):
            self.controller_debug = {"kind": "neural", "observations": inputs,
                                     "parameters": list(self.parameters),
                                     "outputs": [float(value) for value in outputs]}
        return int(np.argmax(outputs)) - 1


def train(population_size, generations, max_frames, seed, visual, worker, visual_speed=4, trials=1):
    random.seed(seed)
    weight_count = worker.request("count")
    if not hasattr(creator, "NeuralFitness"):
        creator.create("NeuralFitness", base.Fitness, weights=(1.0,))
        creator.create("NeuralIndividual", list, fitness=creator.NeuralFitness)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.NeuralIndividual,
                     lambda: [random.uniform(-1, 1) for _ in range(weight_count)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def bounded(operator):
        def wrapped(*args, **kwargs):
            offspring = operator(*args, **kwargs)
            for individual in offspring:
                individual[:] = [max(-5.0, min(5.0, weight)) for weight in individual]
            return offspring
        return wrapped

    toolbox.decorate("mate", bounded)
    toolbox.decorate("mutate", bounded)
    game = NeuralGame([], worker)
    game.training_visual = visual
    game.visual_speed = visual_speed
    game.training_without_graphics = not visual
    game.set_visible(visual)
    seeds = [seed + i for i in range(trials)]
    try:
        population = toolbox.population(n=population_size)
        best = tools.HallOfFame(1)
        print("gen  avaluacions  mitjana  millor", flush=True)
        for generation in range(generations + 1):
            if generation:
                elites = [toolbox.clone(individual) for individual in tools.selBest(population, 2)]
                selected = toolbox.select(population, len(population) - len(elites))
                population[:] = elites + algorithms.varAnd(selected, toolbox, cxpb=0.5, mutpb=0.7)
            worker.reset()
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
            best.update(population)
            scores = [individual.fitness.values[0] for individual in population]
            game.training_history.append((sum(scores) / len(scores), max(scores)))
            print(f"{generation:3}  {evaluations:11}  "
                  f"{sum(scores) / len(scores):7.1f}  {max(scores):6.1f}", flush=True)
        validation_seeds = [seed + 10000 + i for i in range(trials)]
        validation = evaluate(game, best[0], validation_seeds, max_frames)[0]
        print(f"Validació amb llavors noves: {validation:.1f} fotogrames de mitjana", flush=True)
        result = {"architecture": [INPUTS, HIDDEN, OUTPUTS], "observation_version": 2,
                  "validation_seeds": validation_seeds, "validation_mean_frames": validation,
                  "weights": list(best[0]),
                  "mean_survival_frames": best[0].fitness.values[0],
                  "seeds": seeds, "max_frames": max_frames}
        MODEL_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Xarxa desada a {MODEL_FILE}")
    finally:
        game.close()


def main():
    parser = parser_entrenament(__doc__)
    args = parser.parse_args()
    valida_opcions(parser, args)
    if args.entrenar:
        try:
            worker = NetworkWorker()
            try:
                train(args.poblacio, args.generacions, args.fotogrames,
                      args.llavor if args.llavor is not None else 42, args.visual, worker,
                      visual_speed=args.velocitat, trials=args.proves)
            finally:
                worker.close()
        except KeyboardInterrupt:
            print("Entrenament interromput; no s'han desat pesos nous.")
        return
    if not MODEL_FILE.exists():
        parser.error("Primer entrena la xarxa amb --entrenar")
    data = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    if data.get("architecture") != [INPUTS, HIDDEN, OUTPUTS]:
        parser.error("L'arquitectura de la xarxa desada no és compatible")
    weights = data.get("weights", [])
    worker = NetworkWorker()
    try:
        if len(weights) != worker.request("count"):
            parser.error("El nombre de pesos desats no és correcte")
        observation_version = data.get("observation_version", 1)
        if observation_version != 2:
            parser.error("El model usa observacions antigues; torna a entrenar-lo amb --entrenar")
        game = NeuralGame(weights, worker, seed=args.llavor,
                          observation_version=observation_version)
        try:
            game.setup()
            game.run()
        finally:
            game.close()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
