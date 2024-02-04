import random

import arcade
import neat

WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 5
GAME = None


class Game(arcade.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "NEAT Fuzzy UFO Game")
        self.conf = None
        self.players = None
        self.projectiles = None
        self.projectile_speed = 8
        self.projectile_frequency = 30
        self.projectile_counter = 0
        self.projectile_spawn_points = [
            ((WIDTH * i) // 8, HEIGHT)
            for i in range(1, 9)
        ]
        self.lives = 1
        self.score = 0

    def setup(self, genomes, config):
        self.conf = config

        self.projectiles = arcade.SpriteList()
        self.players = arcade.SpriteList()
        for genome_id, genome in genomes:
            player = UFO("images/ufo_green.png", 1, genome, self)
            self.players.append(player)

        arcade.set_background_color(arcade.color.BLACK)
        self.lives = 1
        self.score = 0

    def on_draw(self):
        arcade.start_render()
        self.players.draw()
        self.projectiles.draw()
        arcade.draw_text(f"Lives: {self.lives}", 10, HEIGHT - 30, arcade.color.WHITE, 14)
        arcade.draw_text(f"Score: {self.score}", WIDTH - 100, HEIGHT - 30, arcade.color.WHITE, 14)

    def on_update(self, delta_time):
        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        self.projectiles.update()
        self.players.update()

        if not self.players:
            self.game_over()

    def create_projectile(self):
        spawn_point = random.choice(self.projectile_spawn_points)
        projectile = Projectile(
            "images/meteor.png", 1, center_x=spawn_point[0], center_y=spawn_point[1], app=self
        )
        self.projectiles.append(projectile)


    def get_closest_projectile(self, player):
        # Obtener el proyectil más cercano al jugador
        closest_projectile = None
        closest_distance = float('inf')

        for projectile in self.projectiles:
            distance = arcade.get_distance_between_sprites(player, projectile)
            if distance < closest_distance:
                closest_projectile = projectile
                closest_distance = distance

        return closest_projectile

    def game_over(self):
        print("Game Over!")
        self.players.clear()
        self.projectiles.clear()

        arcade.exit()


class UFO(arcade.Sprite):
    def __init__(self, filename, scale, genome, app):
        super().__init__(filename, scale)
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, app.conf)
        self.app = app
        self.center_x = WIDTH // 2
        self.bottom = 10

    def update(self):
        self.move_player()

        if arcade.check_for_collision_with_list(self, self.app.projectiles):
            self.genome.fitness = self.app.score
            self.kill()

    def move_player(self):
        # Obtener la posición del proyectil más cercano
        closest_projectile = self.app.get_closest_projectile(self)
        projectile_position = (closest_projectile.center_x, closest_projectile.bottom) if closest_projectile else None
        if projectile_position:
            # Utilizar la red neuronal para tomar decisiones
            output = self.net.activate((self.center_x, self.center_y, *projectile_position))
            if output[0] > 0.1:
                self.change_x = PLAYER_SPEED
            elif output[0] < -0.1:
                self.change_x = -PLAYER_SPEED
            else:
                self.change_x = 0

            self.center_x += self.change_x
            self.left = max(0, self.left)
            self.right = min(WIDTH, self.right)


class Projectile(arcade.Sprite):
    def __init__(self, filename, scale, center_x=0, center_y=0, app=None):
        super().__init__(filename, scale)
        self.center_x = center_x
        self.center_y = center_y
        self.change_y = 5
        self.app = app

    def update(self):
        self.center_y -= self.change_y
        if self.top < 0:
            self.kill()
            self.app.score += 1


def eval_genomes(genomes, config):
    global GAME
    GAME.setup(genomes, config)
    arcade.run()


def run_neat():
    global GAME

    config_path = "config-file.txt"  # Reemplaza con la ruta de tu archivo de configuración NEAT
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    GAME = Game()
    winner = population.run(eval_genomes, 10)  # Ajusta el número de generaciones según tus necesidades

    # Puedes hacer lo que quieras con el ganador, como guardarlo en un archivo para su uso posterior
    print("Best genome:\n", winner)


if __name__ == "__main__":
    run_neat()
