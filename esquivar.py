import random

import arcade

from joc_comu import (
    HEIGHT, PLAYER_SPEED, PROJECTILE_FREQUENCY, ROOT, SCALE, SCREEN_HEIGHT,
    SCREEN_WIDTH, WIDTH, create_projectile, parse_seed,
)

FULL_SCREEN = False
BIG_FONT = 48 * SCALE
MED_FONT = 24 * SCALE
SMALL_FONT = 22 * SCALE
CONTROLS = (
    (arcade.key.A, arcade.key.S),
    (arcade.key.F, arcade.key.G),
    (arcade.key.J, arcade.key.K),
    (arcade.key.LEFT, arcade.key.RIGHT),
)
JOYSTICK_DEAD_ZONE = 0.15


class Game(arcade.Window):
    #players: List["UFO"]
    #projectiles: arcade.SpriteList
    #projectile_speed: int
    #projectile_frequency: int
    #projectile_counter: int
    #projectile_spawn_points: List[tuple[int, int]]
    #game_over: bool
    #high_score: int

    def __init__(self, seed=None):
        self.seed = seed
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
        self.projectile_frequency = PROJECTILE_FREQUENCY
        self.projectile_counter = 0
        self.game_over = None
        self.score = 0
        self.high_score = 0

        self.joysticks = []
        self.pressed_keys = set()

    def setup(self):
        self.switch_to()
        self.meteor_rng = random.Random(self.seed)
        # Les llistes diferides no pugen textures a OpenGL fins que es dibuixen.
        lazy = getattr(self, "training_without_graphics", False)
        self.projectiles = arcade.SpriteList(lazy=lazy)
        self.visible_players = arcade.SpriteList(lazy=lazy)
        self.players[0] = UFO(str(ROOT / "images/ufoGreen.png"), SCALE, arcade.color.GREEN)
        self.players[1] = UFO(str(ROOT / "images/ufoRed.png"), SCALE, arcade.color.RED)
        self.players[2] = UFO(str(ROOT / "images/ufoBlue.png"), SCALE, arcade.color.BLUE_GRAY)
        self.players[3] = UFO(str(ROOT / "images/ufoYellow.png"), SCALE, arcade.color.YELLOW)
        self.visible_players.extend(self.players)

        self.game_over = False
        self.pressed_keys.clear()

        self.score = 0

        try:
            with open(ROOT / "high_score.txt") as f:
                self.high_score = int(f.read())
        except FileNotFoundError:
            self.high_score = 0

        if not self.joysticks:
            for joystick in arcade.get_joysticks()[:len(self.players)]:
                joystick.open()
                joystick.push_handlers(self)
                self.joysticks.append(joystick)
        self.projectile_counter = 0
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        self.clear()

        self.visible_players.draw()

        self.projectiles.draw()

        for i in range(4):
            arcade.draw_text(
                f"Player {i + 1}",
                70 + 250 * SCALE * i,
                HEIGHT - 50 * SCALE,
                self.players[i].color,
                MED_FONT,
            )
            #arcade.draw_text(
            #    f"Score: {self.players[i].score}",
            #    70 + 250 * SCALE * i,
            #    HEIGHT - 85 * SCALE,
            #    arcade.color.WHITE,
            #    MED_FONT,
            #)

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
        self.switch_to()
        if self.game_over:
            return

        self.projectile_counter += 1

        if self.projectile_counter == self.projectile_frequency:
            self.create_projectile()
            self.projectile_counter = 0

        self.projectiles.update()
        for projectile in list(self.projectiles):
            if projectile.top < 0:
                projectile.kill()
                self.score += 1
                for i, player in enumerate(self.players):
                    if player.alive:
                        player.score += 1

        for i, player in enumerate(self.players):
            if not player.alive:
                continue
            player.change_x = self.player_direction(i) * PLAYER_SPEED
            player.update()

            if arcade.check_for_collision_with_list(player, self.projectiles, method=3):
                player.kill()
                player.alive = False

        if all([not player.alive for player in self.players]):
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
                with open(ROOT / "high_score.txt", "w") as f:
                    f.write(str(self.score))

    def player_direction(self, index):
        left, right = CONTROLS[index]
        direction = int(right in self.pressed_keys) - int(left in self.pressed_keys)
        if not direction and index < len(self.joysticks):
            axis = self.joysticks[index].x
            if abs(axis) > JOYSTICK_DEAD_ZONE:
                return axis
        return direction

    def create_projectile(self):
        self.projectiles.append(create_projectile(self.meteor_rng, self.score))

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            arcade.exit()
        elif self.game_over:
            self.setup()
        else:
            self.pressed_keys.add(key)

    def on_key_release(self, key, modifiers):
        self.pressed_keys.discard(key)

    def on_joybutton_press(self, _joystick, button):
        if self.game_over:
            self.setup()


class UFO(arcade.Sprite):
    #center_x: int
    #bottom: int
    #alive: bool
    #score: int
    #color: arcade.color

    def __init__(self, filename, scale, color):
        super().__init__(filename, scale)
        self.center_x = WIDTH // 2
        self.bottom = 10 * SCALE
        self.alive = True
        self.score = 0
        self.color = color

    def update(self, delta_time=1 / 60):
        self.move_player()

    def move_player(self):
        self.center_x += self.change_x
        self.left = max(0, self.left)
        self.right = min(WIDTH, self.right)



if __name__ == "__main__":
    game = Game(seed=parse_seed())
    game.setup()
    game.run()
