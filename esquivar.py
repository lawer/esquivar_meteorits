import random
import arcade
import neat

WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 5


class Game(arcade.Window):
    def __init__(self, genome, config):
        super().__init__(WIDTH, HEIGHT, "NEAT Fuzzy UFO Game")
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.player = arcade.Sprite("images/ufo_green.png")
        self.player.center_x = WIDTH // 2
        self.player.bottom = 10
        self.projectiles = arcade.SpriteList()
        self.projectile_speed = 8
        self.projectile_frequency = 30
        self.projectile_counter = 0
        self.projectile_spawn_points = [
            ((WIDTH * i) // 8, HEIGHT)
            for i in range(1, 9)
        ]
        self.lives = 3
        self.score = 0

    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        arcade.start_render()
        self.player.draw()
        self.projectiles.draw()
        arcade.draw_text(f"Lives: {self.lives}", 10, HEIGHT - 30, arcade.color.WHITE, 14)
        arcade.draw_text(f"Score: {self.score}", WIDTH - 100, HEIGHT - 30, arcade.color.WHITE, 14)

    def on_update(self, delta_time):
        self.move_player()
        self.update_projectiles()
        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

    def move_player(self):
        # Obtener la posición del proyectil más cercano
        closest_projectile = self.get_closest_projectile()
        projectile_position = (closest_projectile.center_x, closest_projectile.center_y) if closest_projectile else (0, 0)

        # Utilizar la red neuronal para tomar decisiones
        output = self.net.activate((self.player.center_x, self.player.center_y, *projectile_position))
        if output[0] > 0.5:
            self.player.change_x = -PLAYER_SPEED
        elif output[1] > 0.5:
            self.player.change_x = PLAYER_SPEED
        else:
            self.player.change_x = 0

        self.player.center_x += self.player.change_x
        self.player.left = max(0, self.player.left)
        self.player.right = min(WIDTH, self.player.right)

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

            if arcade.check_for_collision(self.player, projectile):
                projectile.kill()
                self.lives -= 1

                if self.lives <= 0:
                    self.game_over()

    def get_closest_projectile(self):
        # Obtener el proyectil más cercano al jugador
        closest_projectile = None
        closest_distance = float('inf')

        for projectile in self.projectiles:
            distance = arcade.get_distance_between_sprites(self.player, projectile)
            if distance < closest_distance:
                closest_projectile = projectile
                closest_distance = distance

        return closest_projectile

    def game_over(self):
        print("Game Over!")
        arcade.close_window()

    def on_key_press(self, key, modifiers):
        pass

    def on_key_release(self, key, modifiers):
        pass


def eval_genomes(genomes, config):
    for _, genome in genomes:
        game = Game(genome, config)
        game.setup()
        arcade.run()
        genome.fitness = game.score  # Ajusta la función de fitness según tus necesidades


def run_neat():
    config_path = "path/to/your/config-file.txt"  # Reemplaza con la ruta de tu archivo de configuración NEAT
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, 10)  # Ajusta el número de generaciones según
