import json
import queue
import random
import threading
import time

import arcade
import neat

from joc_comu import (
    HEIGHT, PLAYER_SPEED, PROJECTILE_FREQUENCY, ROOT, SCALE, SCREEN_HEIGHT,
    SCREEN_WIDTH, WIDTH, create_projectile, observe,
)

from esquivar import Game as ManualGame
from panell_entrenament import TrainingPanel
from opcions_entrenament import parser_entrenament, valida_opcions

MODEL_FILE = ROOT / "neat_genetic.json"
GAME = None

class Game(arcade.Window):
    def __init__(self, seed=None, visual=True, visual_speed=4, max_frames=1800, trials=1):
        self.seed = seed
        self.visual = visual
        self.max_frames = max_frames
        self.trials = trials
        self.trial = 0
        super().__init__(WIDTH, HEIGHT, "NEAT UFO Game", antialiasing=False,
                         visible=visual, update_rate=1 / (60 * visual_speed))
        self.set_location((SCREEN_WIDTH - WIDTH) // 2, (SCREEN_HEIGHT - HEIGHT) // 2)
        self.conf = None
        self.players = None
        self.projectiles = None
        self.projectile_frequency = PROJECTILE_FREQUENCY
        self.projectile_counter = 0
        self.score = 0
        self.generation = 0
        self.frame = 0
        self.training_history = []
        self.training_fitness_unit = "fot."
        self.training_status = ""
        self.training_snapshot = None
        self.training_panel = None
        self.training_panel_visible = True
        self.inspected_player = None
        self.generation_requests = queue.Queue()
        self.generation_finished = threading.Event()
        self.training_cancelled = threading.Event()
        self.generation_active = False
        self.training_result = None
        self.training_error = None

    def setup(self, genomes, config):
        self.conf = config
        self.meteor_rng = random.Random(self.seed)

        self.projectiles = arcade.SpriteList(lazy=not self.visual)
        self.players = arcade.SpriteList(lazy=not self.visual)
        for genome_id, genome in genomes:
            player = UFO(str(ROOT / "images/ufoGreen.png"), SCALE, genome, self)
            self.players.append(player)

        arcade.set_background_color(arcade.color.BLACK)
        self.score = 0
        self.projectile_counter = 0
        self.frame = 0
        self.training_snapshot = None
        self.inspected_player = None
        self.generation_active = True

    def on_draw(self):
        self.clear()
        if self.players is not None:
            self.players.draw()
        if self.projectiles is not None:
            self.projectiles.draw()
        if self.inspected_player and self.inspected_player in self.players:
            player = self.inspected_player
            arcade.draw_circle_outline(player.center_x, player.center_y,
                                       max(player.width, player.height) * 0.65,
                                       arcade.color.YELLOW, 3 * SCALE)
        if self.training_panel_visible:
            if self.training_panel is None:
                self.training_panel = TrainingPanel()
            self.training_panel.draw(self)

    def on_update(self, delta_time):
        self.process_training_requests()
        if not self.generation_active:
            return
        self.frame += 1
        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        self.projectiles.update()
        for projectile in list(self.projectiles):
            if projectile.top < 0:
                projectile.kill()
                self.score += 1

        self.inspected_player = self.players[0] if self.players else None
        self.players.update()
        for player in list(self.players):
            if arcade.check_for_collision_with_list(player, self.projectiles):
                player.genome.fitness = self.frame
                player.kill()

        self.training_status = (f"Gen. {self.generation}  Prova {self.trial}/{self.trials}  "
                                f"Naus {len(self.players)}  Fot. {self.frame}")
        if self.frame >= self.max_frames:
            for player in list(self.players):
                player.genome.fitness = self.frame
                player.kill()
        if not self.players:
            self.game_over()


    def create_projectile(self):
        self.projectiles.append(create_projectile(self.meteor_rng, self.score))

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.cancel_training()
            arcade.exit()
        elif key == arcade.key.P:
            self.training_panel_visible = not self.training_panel_visible

    def cancel_training(self):
        self.training_cancelled.set()
        self.generation_finished.set()

    def on_close(self):
        self.cancel_training()
        super().on_close()

    def request_generation(self, genomes, config, seed, trial):
        if self.training_cancelled.is_set():
            raise TrainingInterrupted
        self.generation_finished.clear()
        self.generation_requests.put(("generation", genomes, config, seed, trial))
        self.generation_finished.wait()
        if self.training_cancelled.is_set():
            raise TrainingInterrupted

    def process_training_requests(self):
        while True:
            try:
                message = self.generation_requests.get_nowait()
            except queue.Empty:
                return
            if message[0] == "generation":
                self.seed, self.trial = message[3], message[4]
                self.setup(message[1], message[2])
            elif message[0] == "finished":
                self.training_result = message[1]
                self.training_error = message[2]
                arcade.exit()

    def game_over(self):
        self.generation_active = False
        self.projectiles.clear()
        self.generation_finished.set()


class TrainingInterrupted(Exception):
    pass


class UFO(arcade.Sprite):
    def __init__(self, filename, scale, genome, app):
        super().__init__(filename, scale)
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, app.conf)
        self.app = app
        self.center_x = WIDTH // 2
        self.bottom = 10 * SCALE

    def update(self, delta_time=1 / 60):
        self.move_player()

    def move_player(self):
        observations = observe(self, self.app.projectiles)
        output = self.net.activate(observations)
        if output[0] > 0.5:
            self.change_x = PLAYER_SPEED
        elif output[0] < -0.5:
            self.change_x = -PLAYER_SPEED
        else:
            self.change_x = 0

        if self is self.app.inspected_player:
            active = [connection for connection in self.genome.connections.values()
                      if connection.enabled]
            connections = sorted(active, key=lambda connection: abs(connection.weight),
                                 reverse=True)[:8]
            self.app.training_snapshot = {
                "kind": "neat", "identifier": self.genome.key,
                "action": "dreta" if self.change_x > 0 else
                          "esquerra" if self.change_x < 0 else "quieta",
                "observations": observations, "output": output[0],
                "nodes": len(self.genome.nodes), "total_connections": len(active),
                "connections": [(connection.key[0], connection.key[1], connection.weight)
                                for connection in connections],
                "network_edges": [(source, node, weight)
                                  for node, _, _, _, _, links in self.net.node_evals
                                  for source, weight in links],
                "network_nodes": [node for node, *_ in self.net.node_evals],
            }

        self.center_x += self.change_x
        self.left = max(0, self.left)
        self.right = min(WIDTH, self.right)



class PanelReporter(neat.reporting.BaseReporter):
    """Actualitza el gràfic després de completar cada generació."""
    def __init__(self, game):
        self.game = game

    def __getstate__(self):
        # Els checkpoints no han de serialitzar la finestra ni el context OpenGL.
        return {"game": None}

    def start_generation(self, generation):
        if self.game is not None:
            self.game.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if self.game is not None:
            scores = [genome.fitness for genome in population.values()]
            self.game.training_history.append((sum(scores) / len(scores), best_genome.fitness))


def evaluate_trials(genomes, config, seeds):
    totals = {genome_id: 0 for genome_id, _ in genomes}
    for trial, seed in enumerate(seeds, start=1):
        GAME.request_generation(genomes, config, seed, trial)
        for genome_id, genome in genomes:
            totals[genome_id] += genome.fitness
    for genome_id, genome in genomes:
        genome.fitness = totals[genome_id] / len(seeds)


def eval_genomes(genomes, config):
    evaluate_trials(genomes, config, GAME.training_seeds)


def run_neat(seed=42, population_size=12, generations=8, max_frames=1800,
             trials=1, visual=False, visual_speed=4):
    global GAME
    random.seed(seed)
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, str(ROOT / "config-file.txt"),
    )
    config.pop_size = population_size
    config.no_fitness_termination = True
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(1))

    GAME = Game(seed=seed, visual=visual, visual_speed=visual_speed,
                max_frames=max_frames, trials=trials)
    GAME.training_seeds = [seed + i for i in range(trials)]
    population.add_reporter(PanelReporter(GAME))

    def evolve():
        try:
            # Igual que els altres modes: població inicial i N generacions.
            winner = population.run(eval_genomes, generations + 1)
            training_fitness = winner.fitness
            evaluate_trials([(winner.key, winner)], config,
                            [seed + 10000 + i for i in range(trials)])
            print(f"Validació amb llavors noves: {winner.fitness:.1f} fotogrames de mitjana", flush=True)
            winner.fitness = training_fitness
        except TrainingInterrupted:
            return
        except Exception as exc:
            GAME.generation_requests.put(("finished", None, exc))
        else:
            GAME.generation_requests.put(("finished", winner, None))

    thread = threading.Thread(target=evolve, name="neat-training", daemon=True)
    thread.start()
    try:
        if visual:
            arcade.run()
        else:
            while thread.is_alive() or not GAME.generation_requests.empty():
                GAME.on_update(1 / 60)
                if not GAME.generation_active:
                    time.sleep(0.001)
    except KeyboardInterrupt:
        print("Entrenament interromput.")
    finally:
        GAME.cancel_training()
        thread.join()
        if not GAME.closed:
            GAME.close()

    if GAME.training_error is not None:
        raise GAME.training_error
    if GAME.training_result is not None:
        save_winner(GAME.training_result)
        print(f"Xarxa desada a {MODEL_FILE}")
        print("Best genome:\n", GAME.training_result)


def save_winner(genome):
    def attributes(gene):
        return {attribute.name: getattr(gene, attribute.name)
                for attribute in gene._gene_attributes}
    data = {
        "observation_version": 2,
        "key": genome.key,
        "fitness": genome.fitness,
        "nodes": [{"key": key, **attributes(node)} for key, node in genome.nodes.items()],
        "connections": [{"key": list(key), "innovation": connection.innovation,
                         **attributes(connection)}
                        for key, connection in genome.connections.items()],
    }
    MODEL_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_winner(config):
    data = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    if data.get("observation_version") != 2:
        raise ValueError("El model NEAT usa observacions incompatibles")
    genome = neat.DefaultGenome(data["key"])
    genome.fitness = data["fitness"]
    for entry in data["nodes"]:
        node = config.genome_config.node_gene_type(entry["key"])
        for attribute in node._gene_attributes:
            setattr(node, attribute.name, entry[attribute.name])
        genome.nodes[node.key] = node
    for entry in data["connections"]:
        connection = config.genome_config.connection_gene_type(
            tuple(entry["key"]), innovation=entry["innovation"])
        for attribute in connection._gene_attributes:
            setattr(connection, attribute.name, entry[attribute.name])
        genome.connections[connection.key] = connection
    return genome


class NeatPlayGame(ManualGame):
    def __init__(self, network, seed=None):
        super().__init__(seed=seed)
        self.network = network

    def player_direction(self, index):
        if index != 0:
            return super().player_direction(index)
        output = self.network.activate(observe(self.players[0], self.projectiles))[0]
        return 1 if output > 0.5 else -1 if output < -0.5 else 0

    def on_key_press(self, key, modifiers):
        if self.game_over or key not in (arcade.key.A, arcade.key.S):
            super().on_key_press(key, modifiers)


def main():
    parser = parser_entrenament(__doc__, neat=True)
    args = parser.parse_args()
    valida_opcions(parser, args)
    if args.entrenar:
        run_neat(seed=args.llavor if args.llavor is not None else 42,
                 population_size=args.poblacio, generations=args.generacions,
                 max_frames=args.fotogrames, trials=args.proves,
                 visual=args.visual, visual_speed=args.velocitat)
        return
    if not MODEL_FILE.exists():
        parser.error("Primer entrena el controlador amb --entrenar")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                str(ROOT / "config-file.txt"))
    genome = load_winner(config)
    game = NeatPlayGame(neat.nn.FeedForwardNetwork.create(genome, config), seed=args.llavor)
    game.setup()
    game.run()


if __name__ == "__main__":
    main()
