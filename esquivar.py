import random
from typing import List

import arcade

FULL_SCREEN = False
BASE_WIDTH = 1024
BASE_HEIGHT = 768

SCREEN_WIDTH = arcade.get_display_size()[0]
SCREEN_HEIGHT = arcade.get_display_size()[1]

# SCALE = 3
SCALE = min(SCREEN_WIDTH // BASE_WIDTH, SCREEN_HEIGHT // BASE_HEIGHT)

PLAYER_SPEED_BASE = 5
PLAYER_SPEED = PLAYER_SPEED_BASE * SCALE
CHANGE_Y_BASE = 5
CHANGE_Y = CHANGE_Y_BASE * SCALE
WIDTH = BASE_WIDTH * SCALE
HEIGHT = BASE_HEIGHT * SCALE
BIG_FONT_BASE = 48
BIG_FONT = BIG_FONT_BASE * SCALE
MED_FONT_BASE = 24
MED_FONT = MED_FONT_BASE * SCALE
SMALL_FONT_BASE = 22
SMALL_FONT = SMALL_FONT_BASE * SCALE
NUM_SPAWN_POINTS = 16


class Game(arcade.Window):
    players: List["UFO|None"]
    projectiles: arcade.SpriteList | None
    projectile_speed: int
    projectile_frequency: int
    projectile_counter: int
    projectile_spawn_points: List[tuple[int, int]]
    game_over: bool | None
    high_score: int

    def __init__(self):
        super().__init__(
            WIDTH,
            HEIGHT,
            "UFO Game",
            antialiasing=True,
            visible=True,
            fullscreen=FULL_SCREEN,
        )

        _left = SCREEN_WIDTH // 2 - WIDTH // 2
        _top = SCREEN_HEIGHT // 2 - HEIGHT // 2
        self.set_location(_left, _top)

        self.players = [None] * 4
        self.projectiles = None
        self.projectile_speed = CHANGE_Y
        self.projectile_frequency = 30
        self.projectile_counter = 0
        self.projectile_spawn_points = [
            ((WIDTH * i) // NUM_SPAWN_POINTS, HEIGHT)
            for i in range(1, NUM_SPAWN_POINTS + 1)
        ]
        self.game_over = None
        self.score = 0
        self.high_score = 0

        self.joysticks = None

    def setup(self):
        self.projectiles = arcade.SpriteList()
        self.players[0] = UFO("images/ufoGreen.png", SCALE, arcade.color.GREEN)
        self.players[1] = UFO("images/ufoRed.png", SCALE, arcade.color.RED)
        self.players[2] = UFO("images/ufoBlue.png", SCALE, arcade.color.BLUE_GRAY)
        self.players[3] = UFO("images/ufoYellow.png", SCALE, arcade.color.YELLOW)

        self.game_over = False

        self.score = 0

        try:
            with open("high_score.txt") as f:
                self.high_score = int(f.read())
        except FileNotFoundError:
            self.high_score = 0

        joysticks = arcade.get_joysticks()
        # If we have any...
        if joysticks:
            for joystick in joysticks:
                self.joysticks.append(joystick)
                joystick.open()
                joystick.push_handlers(self)
        else:
            self.joysticks = []
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        arcade.start_render()

        for player in self.players:
            player.draw()

        self.projectiles.draw()

        for i in range(4):
            arcade.draw_text(
                f"Player {i + 1}",
                70 + 250 * SCALE * i,
                HEIGHT - 50 * SCALE,
                self.players[i].color,
                MED_FONT,
            )
            arcade.draw_text(
                f"Score: {self.players[i].score}",
                70 + 250 * SCALE * i,
                HEIGHT - 85 * SCALE,
                arcade.color.WHITE,
                MED_FONT,
            )

        arcade.draw_text(
            f"High Score: {self.high_score}", 50, 50, arcade.color.WHITE, SMALL_FONT
        )

        if self.game_over:
            arcade.draw_text(
                "Game Over!",
                WIDTH // 2,
                HEIGHT // 2 + 80 * SCALE,
                arcade.color.RED,
                BIG_FONT,
                anchor_x="center",
            )
            # Show the winner, his score, his color and the other players' scores
            winner = max(self.players, key=lambda player: player.score)
            winner_pos = self.players.index(winner) + 1
            arcade.draw_text(
                f"Player {winner_pos} wins!",
                WIDTH // 2,
                HEIGHT // 2 - 10 * SCALE,
                winner.color,
                BIG_FONT,
                anchor_x="center",
            )

            for i, player in enumerate(self.players):
                arcade.draw_text(
                    f"Player {i + 1}: {player.score}",
                    WIDTH // 2,
                    HEIGHT // 2 - 80 * SCALE - 50 * SCALE * i,
                    player.color,
                    MED_FONT,
                    anchor_x="center",
                )

    def on_update(self, delta_time):
        global CHANGE_Y

        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        self.projectiles.update()
        for projectile in self.projectiles:
            if projectile.top < 0:
                projectile.kill()
                self.score += 1
                for i, player in enumerate(self.players):
                    if player.alive:
                        player.score += 1

                    projectile.kill()

                if self.score % 10 == 0:
                    self.projectile_speed += 1
                    self.projectile_frequency -= 1

        for player in self.players:
            player.update()

            if arcade.check_for_collision_with_list(player, self.projectiles):
                player.kill()
                player.alive = False

        if all([not player.alive for player in self.players]):
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
                with open("high_score.txt", "w") as f:
                    f.write(str(self.score))

        for i, joystick in enumerate(self.joysticks):
            self.players[i].change_x = joystick.x * PLAYER_SPEED

    def create_projectile(self):
        spawn_point = random.choice(self.projectile_spawn_points)
        image_index = random.randint(1, 4)
        image_name = f"images/meteorBrown_big{image_index}.png"
        projectile = Projectile(
            image_name,
            SCALE,
            center_x=spawn_point[0],
            center_y=spawn_point[1],
            projectile_speed=self.projectile_speed,
        )
        self.projectiles.append(projectile)

    def game_over(self):
        print("Game Over!")
        self.projectiles.clear()

        arcade.exit()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            arcade.exit()
        elif self.game_over:
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
    center_x: int
    bottom: int
    alive: bool
    score: int
    color: arcade.color

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
    center_x: float
    center_y: float
    change_y: float

    def __init__(
        self, filename, scale, center_x=0, center_y=0, projectile_speed=CHANGE_Y
    ):
        super().__init__(filename, scale)
        self.center_x = center_x
        self.center_y = center_y
        self.change_y = projectile_speed

    def update(self):
        self.center_y -= self.change_y


if __name__ == "__main__":
    game = Game()
    game.setup()
    game.run()
