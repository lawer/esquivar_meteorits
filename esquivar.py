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
        self.players = []
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
        for genome_id, genome in genomes:
            player = UFO("images/ufo_green.png", 1, genome, self)
            self.players.append(player)

        arcade.set_background_color(arcade.color.BLACK)
        self.lives = 1
        self.score = 0

    def on_draw(self):
        arcade.start_render()
        for player in self.players:
            player.draw()
        self.projectiles.draw()
        arcade.draw_text(f"Lives: {self.lives}", 10, HEIGHT - 30, arcade.color.WHITE, 14)
        arcade.draw_text(f"Score: {self.score}", WIDTH - 100, HEIGHT - 30, arcade.color.WHITE, 14)

    def on_update(self, delta_time):
        self.update_projectiles()
        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        for player in self.players:
            player.update()

    def create_projectile(self):
        spawn_point = random.choice(self.projectile_spawn_points)
        projectile = arcade.Sprite("images/meteor.png", center_x=spawn_point[0], center_y=spawn_point[1])
        self.projectiles.append(projectile)

    def update_projectiles(self):
        for projectile in self.projectiles:
            projectile.center_y -= self.projectile_speed

            if projectile.top < 0:
                projectile.kill()
                self.score += 1

            for player in self.players:
                if arcade.check_for_collision(player, projectile):
                    #projectile.kill()
                    player.genome.fitness = self.score
                    self.players.remove(player)
                    player.kill()

                    if not self.players:
                        self.game_over()

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
        # Obtener la posición del proyectil más cercano
        closest_projectile = self.app.get_closest_projectile(self)
        projectile_position = (closest_projectile.center_x, closest_projectile.center_y) if closest_projectile else (
            0, 0)

        # Utilizar la red neuronal para tomar decisiones
        output = self.net.activate((self.center_x, self.center_y, *projectile_position))
        if output[0] > 0:
            self.change_x = PLAYER_SPEED
        elif output[0] < 0:
            self.change_x = -PLAYER_SPEED
        else:
            self.change_x = 0

        self.center_x += self.change_x
        self.left = max(0, self.left)
        self.right = min(WIDTH, self.right)


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
