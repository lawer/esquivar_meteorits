import random

import arcade

WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 5


class Game(arcade.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Fuzzy UFO Game")
        self.player = arcade.Sprite("images/ufo_green.png")
        self.player.center_x = WIDTH // 2
        self.player.bottom = 10
        self.projectiles = arcade.SpriteList()
        self.projectile_speed = 8
        self.projectile_frequency = 30  # Aparece aproximadamente una vez por segundo
        self.projectile_counter = 0  # Agregamos el contador de proyectiles
        self.projectile_spawn_points = [
            ((WIDTH * i) // 8, HEIGHT)
            for i in range(1, 9)
        ]
        self.lives = 3
        self.score = 0  # Agregamos una variable para llevar la cuenta de los puntos

    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        arcade.start_render()
        self.player.draw()
        self.projectiles.draw()

        # Mostrar vidas y puntos en la pantalla
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
                self.score += 1  # Incrementar el puntaje al esquivar un proyectil

            if arcade.check_for_collision(self.player, projectile):
                self.handle_collision(projectile)

    def handle_collision(self, projectile):
        projectile.kill()
        self.lives -= 1

        if self.lives <= 0:
            self.game_over()

    def game_over(self):
        print("Game Over!")
        arcade.close_window()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.player.change_x = -PLAYER_SPEED
        elif key == arcade.key.RIGHT:
            self.player.change_x = PLAYER_SPEED

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player.change_x = 0


if __name__ == "__main__":
    game = Game()
    game.setup()
    arcade.run()
