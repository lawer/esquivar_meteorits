import random

import pgzrun

WIDTH = 800
HEIGHT = 600

player = Actor("ufo_green")
player.midbottom = (WIDTH // 2, HEIGHT - 10)

projectiles = []
projectile_speed = 5
projectile_frequency = 50
projectile_counter = 0


def draw():
    screen.clear()
    player.draw()
    for projectile in projectiles:
        projectile.draw()


def update():
    global projectile_counter

    move_player()
    update_projectiles()

    projectile_counter += 1
    if projectile_counter == projectile_frequency:
        create_projectile()
        projectile_counter = 0


def move_player():
    if keyboard.left:
        player.x -= 5
    elif keyboard.right:
        player.x += 5

    # Limitar el movimiento del jugador dentro de la pantalla
    player.left = max(0, player.left)
    player.right = min(WIDTH, player.right)


def create_projectile():
    projectile = Actor("meteor")
    projectile.midbottom = (random.randint(0, WIDTH), 0)
    projectiles.append(projectile)


def update_projectiles():
    global projectiles

    for projectile in projectiles:
        projectile.y += projectile_speed

        # Verificar si el jugador es golpeado por un proyectil
        if projectile.colliderect(player):
            game_over()

        # Si el proyectil sale de la pantalla, eliminarlo
        if projectile.y > HEIGHT:
            projectiles.remove(projectile)


def game_over():
    print("Game Over!")
    exit()


pgzrun.go()
