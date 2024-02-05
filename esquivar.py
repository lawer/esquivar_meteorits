import random

import arcade

WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 5


class Game(arcade.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "UFO Game", antialiasing=False, visible=True)
        self.player = None
        self.projectiles = None
        self.projectile_speed = 8
        self.projectile_frequency = 30
        self.projectile_counter = 0
        self.projectile_spawn_points = [
            ((WIDTH * i) // 8, HEIGHT)
            for i in range(1, 9)
        ]
        self.score = 0

    def setup(self):
        self.projectiles = arcade.SpriteList()
        self.player = UFO("images/ufo_green.png", 1)
        arcade.set_background_color(arcade.color.BLACK)
        self.score = 0

    def on_draw(self):
        arcade.start_render()
        self.player.draw()
        self.projectiles.draw()
        arcade.draw_text(f"Score: {self.score}", WIDTH - 100, HEIGHT - 30, arcade.color.WHITE, 14)

    def on_update(self, delta_time):
        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        self.projectiles.update()
        for projectile in self.projectiles:
            if projectile.top < 0:
                projectile.kill()

        self.player.update()
        if arcade.check_for_collision_with_list(self.player, self.projectiles):
            self.player.kill()
            self.game_over()

    def create_projectile(self):
        spawn_point = random.choice(self.projectile_spawn_points)
        projectile = Projectile(
            "images/meteor.png", 1, center_x=spawn_point[0], center_y=spawn_point[1]
        )
        self.projectiles.append(projectile)

    def game_over(self):
        print("Game Over!")
        self.projectiles.clear()

        arcade.exit()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.player.change_x = -PLAYER_SPEED
        elif key == arcade.key.RIGHT:
            self.player.change_x = PLAYER_SPEED

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player.change_x = 0


class UFO(arcade.Sprite):
    def __init__(self, filename, scale):
        super().__init__(filename, scale)
        self.center_x = WIDTH // 2
        self.bottom = 10

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
