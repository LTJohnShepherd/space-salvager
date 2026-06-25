"""Manual mode: human controls the ship with arrow keys / WASD.

Hold a direction key to move; release for momentum to stop only on key changes.
Press R to reset after game over / win, ESC or window close to quit.
"""

import pygame
from game.space_game import SpaceGame


def main():
    game = SpaceGame(render=True, fps=60)
    game.overlay_lines = ["Manual Mode", "Arrows/WASD = move", "R = reset", "M = mute"]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_m:
                    game.toggle_mute()

        # continuous key state -> movement (manual control).
        # Independent axes so two keys held = diagonal movement.
        keys = pygame.key.get_pressed()
        dx = dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1
        game.set_velocity(dx, dy)

        if not game.is_done():
            game.update()
        game.render()

    game.close()


if __name__ == "__main__":
    main()
