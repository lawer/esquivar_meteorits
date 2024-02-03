import multiprocessing
import random
import arcade
import neat
from multiprocessing import Pool

WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 5


class Game(arcade.Window):
    def __init__(self, genome, config):
        super().__init__(WIDTH, HEIGHT, "NEAT Fuzzy UFO Game")
        self.genome = genome
        self.conf = config
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
        self.lives = 1
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
        closest_projectile = self.get_closest_projectile()
        projectile_position = (closest_projectile.center_x, closest_projectile.bottom) if closest_projectile else None

        if closest_projectile:
            output = self.net.activate((self.player.center_x, self.player.center_y, *projectile_position))
            if output[0] < -0.1 and self.player.left > 0:
                self.player.change_x = -PLAYER_SPEED
            elif output[0] > 0.1 and self.player.right < WIDTH:
                self.player.change_x = PLAYER_SPEED
            else:
                self.player.change_x = 0

        self.player.update()


    def create_projectile(self):
        spawn_point = random.choice(self.projectile_spawn_points)
        projectile = arcade.Sprite("images/meteor.png", center_x=spawn_point[0], center_y=spawn_point[1])
        self.projectiles.append(projectile)

    def update_projectiles(self):
        closest_projectile = self.get_closest_projectile()

        for projectile in self.projectiles:
            projectile.center_y -= self.projectile_speed

            if projectile.top < 0:
                projectile.kill()

                if closest_projectile == projectile:
                    self.score += 1

            if arcade.check_for_collision(self.player, projectile):
                projectile.kill()
                self.lives -= 1

                if self.lives <= 0:
                    self.game_over()

    def get_closest_projectile(self):
        closest_projectile = None
        closest_distance = float('inf')

        for projectile in self.projectiles:
            distance = arcade.get_distance_between_sprites(self.player, projectile)
            if distance < closest_distance:
                closest_projectile = projectile
                closest_distance = distance

        return closest_projectile

    def game_over(self):
        print("Game Over! Score:", self.score)
        arcade.close_window()
        arcade.exit()

    def on_key_press(self, key, modifiers):
        pass

    def on_key_release(self, key, modifiers):
        pass


def eval_genome(genome, config):
    game = Game(genome, config)
    game.setup()
    arcade.run()
    return game.score


def run_neat():
    config_path = "config-file.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(1))

    pe = neat.ParallelEvaluator(20, eval_genome)

    winner = population.run(pe.evaluate, 100)

    print("Best genome:\n", winner)


if __name__ == "__main__":
    run_neat()
