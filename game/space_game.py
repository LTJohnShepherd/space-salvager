"""Space Salvager - pure game logic + rendering.

Single coherent game object driven identically by all control modes
(manual keyboard, camera/hand, trained AI). The RL environment in
rl/space_env.py wraps this same class.

Public contract (used by every entry point and the Gym env):
    reset()
    move_left() / move_right() / move_up() / move_down() / idle()
    update()
    get_state()  -> list[float]   (normalized 0..1, length 6)
    get_reward() -> float
    is_done()    -> bool
    render()
    close()
"""

import os
import math
import random
import pygame

# ---- World / timing ----
WIDTH = 800
HEIGHT = 600
PLAYER_SPEED = 6.0          # pixels per step
PIRATE_BASE_SPEED = 2.2     # pixels per step (scaled by difficulty)
PLAYER_RADIUS = 22
ORE_RADIUS = 18
PIRATE_RADIUS = 26
NUM_PIRATES = 3
STEP_LIMIT = 2000           # truncation
TARGET_SCORE = 15           # success condition

# ---- Colors (fallback if images fail to load) ----
BLACK = (6, 10, 20)
WHITE = (235, 240, 255)
GREEN = (80, 255, 190)
GOLD = (255, 200, 50)
RED = (255, 90, 90)

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
ORE_IMAGES = ["AsteroidRUAOre.png", "AsteroidRUBOre.png",
              "AsteroidRUCOre.png", "AsteroidRUMOre.png"]


def _load(path, size):
    """Load+scale an image, returning None on any failure (game still runs)."""
    try:
        img = pygame.image.load(path)
        img = img.convert_alpha() if pygame.display.get_surface() else img
        return pygame.transform.smoothscale(img, size)
    except Exception:
        return None


class SpaceGame:
    def __init__(self, render: bool = True, fps: int = 60):
        self.render_enabled = render
        self.fps = fps

        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Space Salvager")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 28)
            self._load_assets()
        else:
            self.screen = None
            self.clock = None
            self.font = None
            self.bg = self.player_img = self.pirate_img = None
            self.ore_imgs = []

        # difficulty is set externally (e.g. DeepFace emotion in camera mode)
        self.difficulty = 1.0
        # background camera frame (optional, set by camera_play)
        self.camera_surface = None
        self.overlay_lines = []
        self.reset()

    # ---------------- asset loading ----------------
    def _load_assets(self):
        img = os.path.join(_ASSETS, "images")
        self.bg = _load(os.path.join(img, "nebula_15.png"), (WIDTH, HEIGHT))
        self.player_img = _load(os.path.join(img, "Interceptor.png"),
                                (PLAYER_RADIUS * 2, PLAYER_RADIUS * 2))
        self.pirate_img = _load(os.path.join(img, "PirateCruiser.png"),
                                (PIRATE_RADIUS * 2, PIRATE_RADIUS * 2))
        self.ore_imgs = [_load(os.path.join(img, n),
                               (ORE_RADIUS * 2, ORE_RADIUS * 2))
                         for n in ORE_IMAGES]
        snd = os.path.join(_ASSETS, "sounds")
        self.sounds = {}
        try:
            pygame.mixer.init()
            for key, fn in (("pickup", "pickup.ogg"),
                            ("crash", "crash.ogg"),
                            ("start", "start.ogg")):
                p = os.path.join(snd, fn)
                if os.path.exists(p):
                    self.sounds[key] = pygame.mixer.Sound(p)
        except Exception:
            self.sounds = {}

    def _play(self, key):
        s = self.sounds.get(key) if hasattr(self, "sounds") else None
        if s is not None:
            try:
                s.play()
            except Exception:
                pass

    # ---------------- lifecycle ----------------
    def reset(self):
        self.player = pygame.math.Vector2(WIDTH / 2, HEIGHT / 2)
        self.vel = pygame.math.Vector2(0, 0)
        self.ore = self._random_ore()
        self.ore_kind = random.randrange(len(ORE_IMAGES))
        self.pirates = [self._random_pirate() for _ in range(NUM_PIRATES)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.won = False
        self.last_reward = 0.0
        self._prev_dist = self._dist(self.player, self.ore)
        self._play("start")
        return self.get_state()

    def close(self):
        if self.render_enabled:
            pygame.quit()

    # ---------------- spawning helpers ----------------
    def _random_ore(self):
        m = 60
        return pygame.math.Vector2(random.randint(m, WIDTH - m),
                                   random.randint(m, HEIGHT - m))

    def _random_pirate(self):
        # spawn at an edge, away from the player, with a random velocity
        edge = random.choice(["t", "b", "l", "r"])
        if edge == "t":
            pos = pygame.math.Vector2(random.randint(0, WIDTH), 0)
        elif edge == "b":
            pos = pygame.math.Vector2(random.randint(0, WIDTH), HEIGHT)
        elif edge == "l":
            pos = pygame.math.Vector2(0, random.randint(0, HEIGHT))
        else:
            pos = pygame.math.Vector2(WIDTH, random.randint(0, HEIGHT))
        ang = random.uniform(0, 2 * math.pi)
        return {"pos": pos, "dir": pygame.math.Vector2(math.cos(ang), math.sin(ang))}

    @staticmethod
    def _dist(a, b):
        return (a - b).length()

    # ---------------- actions ----------------
    def move_left(self):  self.vel = pygame.math.Vector2(-PLAYER_SPEED, 0)
    def move_right(self): self.vel = pygame.math.Vector2(PLAYER_SPEED, 0)
    def move_up(self):    self.vel = pygame.math.Vector2(0, -PLAYER_SPEED)
    def move_down(self):  self.vel = pygame.math.Vector2(0, PLAYER_SPEED)
    def idle(self):       self.vel = pygame.math.Vector2(0, 0)

    # ---------------- step ----------------
    def update(self):
        if self.game_over:
            return

        self.steps += 1

        # move player, clamp to screen
        self.player += self.vel
        self.player.x = max(PLAYER_RADIUS, min(WIDTH - PLAYER_RADIUS, self.player.x))
        self.player.y = max(PLAYER_RADIUS, min(HEIGHT - PLAYER_RADIUS, self.player.y))

        # move pirates (speed scaled by difficulty), bounce off walls
        pspeed = PIRATE_BASE_SPEED * self.difficulty
        for p in self.pirates:
            p["pos"] += p["dir"] * pspeed
            if p["pos"].x < 0 or p["pos"].x > WIDTH:
                p["dir"].x *= -1
            if p["pos"].y < 0 or p["pos"].y > HEIGHT:
                p["dir"].y *= -1
            p["pos"].x = max(0, min(WIDTH, p["pos"].x))
            p["pos"].y = max(0, min(HEIGHT, p["pos"].y))

        reward = -0.01  # small step penalty

        # distance shaping (reward variant B - toggle for the report)
        dist = self._dist(self.player, self.ore)
        reward += 0.02 if dist < self._prev_dist else -0.005
        self._prev_dist = dist

        # ore pickup
        if dist < PLAYER_RADIUS + ORE_RADIUS:
            self.score += 1
            reward += 1.0
            self._play("pickup")
            self.ore = self._random_ore()
            self.ore_kind = random.randrange(len(ORE_IMAGES))
            self._prev_dist = self._dist(self.player, self.ore)
            if self.score >= TARGET_SCORE:
                self.won = True
                self.game_over = True

        # pirate collision
        for p in self.pirates:
            if self._dist(self.player, p["pos"]) < PLAYER_RADIUS + PIRATE_RADIUS:
                reward = -1.0
                self.game_over = True
                self._play("crash")
                break

        # truncation
        if self.steps >= STEP_LIMIT:
            self.game_over = True

        self.last_reward = reward

    # ---------------- RL contract ----------------
    def get_state(self):
        """Normalized observation, length 6. Nearest pirate only."""
        nearest = min(self.pirates,
                      key=lambda p: self._dist(self.player, p["pos"]))
        return [
            self.player.x / WIDTH, self.player.y / HEIGHT,
            self.ore.x / WIDTH, self.ore.y / HEIGHT,
            nearest["pos"].x / WIDTH, nearest["pos"].y / HEIGHT,
        ]

    def get_reward(self):
        return float(self.last_reward)

    def is_done(self):
        return self.game_over

    # ---------------- rendering ----------------
    def render(self):
        if not self.render_enabled:
            return
        pygame.event.pump()

        # background: camera frame (camera mode) > nebula > solid fill
        if self.camera_surface is not None:
            self.screen.blit(self.camera_surface, (0, 0))
        elif self.bg is not None:
            self.screen.blit(self.bg, (0, 0))
        else:
            self.screen.fill(BLACK)

        # ore
        ore_img = self.ore_imgs[self.ore_kind] if self.ore_imgs else None
        if ore_img is not None:
            self.screen.blit(ore_img, (self.ore.x - ORE_RADIUS, self.ore.y - ORE_RADIUS))
        else:
            pygame.draw.circle(self.screen, GOLD, (int(self.ore.x), int(self.ore.y)), ORE_RADIUS)

        # pirates
        for p in self.pirates:
            if self.pirate_img is not None:
                self.screen.blit(self.pirate_img,
                                 (p["pos"].x - PIRATE_RADIUS, p["pos"].y - PIRATE_RADIUS))
            else:
                pygame.draw.circle(self.screen, RED,
                                   (int(p["pos"].x), int(p["pos"].y)), PIRATE_RADIUS)

        # player
        if self.player_img is not None:
            self.screen.blit(self.player_img,
                             (self.player.x - PLAYER_RADIUS, self.player.y - PLAYER_RADIUS))
        else:
            pygame.draw.circle(self.screen, GREEN,
                               (int(self.player.x), int(self.player.y)), PLAYER_RADIUS)

        # HUD
        y = 8
        for line in ([f"Score: {self.score}/{TARGET_SCORE}",
                      f"Difficulty: {self.difficulty:.2f}"] + self.overlay_lines[:6]):
            shadow = self.font.render(line, True, BLACK)
            surf = self.font.render(line, True, WHITE)
            self.screen.blit(shadow, (11, y + 1))
            self.screen.blit(surf, (10, y))
            y += 24

        if self.game_over:
            msg = "YOU WIN!  (R to restart)" if self.won else "GAME OVER  (R to restart)"
            surf = self.font.render(msg, True, GOLD if self.won else RED)
            self.screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        self.clock.tick(self.fps)
