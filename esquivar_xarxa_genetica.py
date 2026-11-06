"""Xarxa neuronal d'AITK amb pesos evolucionats amb el GA d'AITK."""

import json
import random
import subprocess
import sys

import numpy as np

from esquivar_fuzzy_adaptatiu import AdaptiveFuzzyGame
from entrenament_comu import evolve
from genetica_comuna import make_population, make_toolbox
from joc_comu import OBSERVATION_VERSION, ROOT, observe
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
    def __init__(self, parameters, worker, seed=None, observation_version=OBSERVATION_VERSION):
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
    toolbox = make_toolbox('neural', weight_count, prefix='Neural')
    game = NeuralGame([], worker)
    game.training_visual = visual
    game.visual_speed = visual_speed
    game.training_without_graphics = not visual
    game.set_visible(visual)
    try:
        population = make_population(toolbox, population_size)
        best, seeds, validation_seeds, validation = evolve(
            game, toolbox, population, seed, trials, generations, max_frames, visual,
            before_generation=worker.reset,
        )
        result = {"architecture": [INPUTS, HIDDEN, OUTPUTS], "observation_version": OBSERVATION_VERSION,
                  "validation_seeds": validation_seeds, "validation_mean_frames": validation,
                  "weights": list(best),
                  "mean_survival_frames": best.fitness.values[0],
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
        if observation_version != OBSERVATION_VERSION:
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
