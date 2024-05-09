import random

import arcade

WIDTH = 1600
HEIGHT = 1200
PLAYER_SPEED = 5


class Game(arcade.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "UFO Game", antialiasing=False, visible=True)
        self.players = [None] * 4

        self.projectiles = None
        self.projectile_speed = 8
        self.projectile_frequency = 30
        self.projectile_counter = 0
        self.projectile_spawn_points = [
            ((WIDTH * i) // 16, HEIGHT)
            for i in range(1, 17)
        ]
        self.scores = [0] * 4
        self.colors = [arcade.color.GREEN, arcade.color.RED, arcade.color.BLUE, arcade.color.YELLOW]

    def setup(self):
        self.projectiles = arcade.SpriteList()

        self.players[0] = UFO("images/ufoGreen.png", 1.5)
        self.players[1] = UFO("images/ufoRed.png", 1.5)
        self.players[2] = UFO("images/ufoBlue.png", 1.5)
        self.players[3] = UFO("images/ufoYellow.png", 1.5)

        arcade.set_background_color(arcade.color.BLACK)

        for score in self.scores:
            self.score = score

    def on_draw(self):
        arcade.start_render()

        for player in self.players:
            player.draw()

        self.projectiles.draw()

        for i in range(4):
            arcade.draw_text(f"Player {i + 1}", 100 + 400 * i, HEIGHT - 50, self.colors[i], 34)
            arcade.draw_text(f"Score: {self.scores[i]}", 100 + 400 * i, HEIGHT - 100, arcade.color.WHITE, 34)

    def on_update(self, delta_time):
        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        self.projectiles.update()
        for projectile in self.projectiles:
            if projectile.top < 0:
                projectile.kill()
                for i, player in enumerate(self.players):
                    if player.alive:
                        self.scores[i] += 1

                    projectile.kill()

        for player in self.players:
            player.update()

            if arcade.check_for_collision_with_list(player, self.projectiles):
                player.kill()
                player.alive = False
                #self.game_over()

    def create_projectile(self):
        spawn_point = random.choice(self.projectile_spawn_points)
        projectile = Projectile(
            "images/meteor.png", 1.5, center_x=spawn_point[0], center_y=spawn_point[1]
        )
        self.projectiles.append(projectile)

    def game_over(self):
        print("Game Over!")
        self.projectiles.clear()

        arcade.exit()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.A:
            self.players[0].change_x = -PLAYER_SPEED
        elif key == arcade.key.S:
            self.players[0].change_x = PLAYER_SPEED
        elif key == arcade.key.F:
            self.players[1].change_x = -PLAYER_SPEED
        elif key == arcade.key.G:
            self.players[1].change_x = PLAYER_SPEED
        elif key == arcade.key.J:
            self.players[2].change_x = -PLAYER_SPEED
        elif key == arcade.key.K:
            self.players[2].change_x = PLAYER_SPEED
        elif key == arcade.key.LEFT:
            self.players[3].change_x = -PLAYER_SPEED
        elif key == arcade.key.RIGHT:
            self.players[3].change_x = PLAYER_SPEED

    def on_key_release(self, key, modifiers):
        if key == arcade.key.A or key == arcade.key.S:
            self.players[0].change_x = 0
        elif key == arcade.key.F or key == arcade.key.G:
            self.players[1].change_x = 0
        elif key == arcade.key.J or key == arcade.key.K:
            self.players[2].change_x = 0
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.players[3].change_x = 0


class UFO(arcade.Sprite):
    def __init__(self, filename, scale):
        super().__init__(filename, scale)
        self.center_x = WIDTH // 2
        self.bottom = 10
        self.alive = True

    def update(self):
        self.move_player()

    def move_player(self):
        self.center_x += self.change_x
        self.left = max(0, self.left)
        self.right = min(WIDTH, self.right)


class Projectile(arcade.Sprite):
    def __init__(self, filename, scale, center_x=0, center_y=0):
        super().__init__(filename, scale)
        self.center_x = center_x
        self.center_y = center_y
        self.change_y = 5

    def update(self):
        self.center_y -= self.change_y


if __name__ == "__main__":
    game = Game()
    game.setup()
    game.run()
