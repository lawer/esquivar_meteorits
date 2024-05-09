import random

import arcade

WIDTH = 800
HEIGHT = 600
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
        self.game_over = None
        self.high_score = 0

        self.joysticks = None

    def setup(self):
        self.projectiles = arcade.SpriteList()
        self.players[0] = UFO("images/ufoGreen.png", 1, arcade.color.GREEN)
        self.players[1] = UFO("images/ufoRed.png", 1, arcade.color.RED)
        self.players[2] = UFO("images/ufoBlue.png", 1, arcade.color.BLUE_GRAY)
        self.players[3] = UFO("images/ufoYellow.png", 1, arcade.color.YELLOW)

        self.game_over = False

        try:
            with open("high_score.txt") as f:
                self.high_score = int(f.read())
        except FileNotFoundError:
            self.high_score = 0

        joysticks = arcade.get_joysticks()
        # If we have any...
        if joysticks:
            self.joysticks = []
            for joystick in joysticks:
                self.joysticks.append(joystick)
                joystick.open()
                joystick.push_handlers(self)

        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        arcade.start_render()

        for player in self.players:
            player.draw()

        self.projectiles.draw()

        for i in range(4):
            arcade.draw_text(f"Player {i + 1}", 50 + 200 * i, HEIGHT - 50, self.players[i].color, 24)
            arcade.draw_text(f"Score: {self.players[i].score}", 50 + 200 * i, HEIGHT - 85, arcade.color.WHITE, 24)

        arcade.draw_text(f"High Score: {self.high_score}", WIDTH - 760, HEIGHT - 550, arcade.color.WHITE, 24)

        if self.game_over:
            arcade.draw_text("Game Over!", WIDTH // 2, HEIGHT // 2 + 20, arcade.color.RED, 48, anchor_x="center")
            # Show the winner, his score, his color and the other players' scores
            winner = max(self.players, key=lambda player: player.score)
            winner_pos = self.players.index(winner) + 1
            arcade.draw_text(f"Player {winner_pos} wins!", WIDTH // 2, HEIGHT // 2 - 50, winner.color, 48, anchor_x="center")

            for i, player in enumerate(self.players):
                arcade.draw_text(f"Player {i + 1}: {player.score}", WIDTH // 2, HEIGHT // 2 - 100 - 50 * i, player.color, 24, anchor_x="center")


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
                        player.score += 1

                    projectile.kill()

        for player in self.players:
            player.update()

            if arcade.check_for_collision_with_list(player, self.projectiles):
                player.kill()
                player.alive = False

                if all([not player.alive for player in self.players]):
                    self.game_over = True
                    high_score = max(player.score for player in self.players)
                    if high_score > self.high_score:
                        self.high_score = high_score
                        with open("high_score.txt", "w") as f:
                            f.write(str(high_score))

        for i, joystick in enumerate(self.joysticks):
            self.players[i].change_x =  joystick.x * PLAYER_SPEED

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
        if self.game_over:
            self.setup()
        else:
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

    def on_joybutton_press(self, _joystick, button):
        if self.game_over:
            self.setup()

class UFO(arcade.Sprite):
    def __init__(self, filename, scale, color):
        super().__init__(filename, scale)
        self.center_x = WIDTH // 2
        self.bottom = 10
        self.alive = True
        self.score = 0
        self.color = color

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
