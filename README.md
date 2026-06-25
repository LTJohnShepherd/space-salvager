# Space Salvager

A 2D space game built with **Pygame** where one ship collects ore while dodging
pirate ships. The same game can be played three ways — by **keyboard**, by
**hand gestures + facial emotion** through a webcam, and by a **reinforcement-learning
agent** that learned to play it on its own.

> **Submitter:** _maksim kif_

---

## 1. Short description
You pilot a single salvager ship in an asteroid field. Fly into ore to collect
it; touch a pirate cruiser and you're destroyed. Reach the target score to win.

## 2. Goal
Collect **15** ore pickups without colliding with a pirate.

## 3. Rules
- Moving into an ore pickup scores +1 and spawns new ore.
- Colliding with any pirate ends the run.
- Pirates bounce around the field; their speed scales with the difficulty.
- The episode also ends on reaching the target score (win) or the step limit.

## 4. Game modes
| Mode | File | Control source |
|------|------|----------------|
| Manual | `manual_play.py` | Keyboard (arrows / WASD) |
| Camera | `camera_play.py` | Hand gesture (MediaPipe) + emotion (DeepFace) |
| AI | `ai_play.py` | Trained PPO model plays by itself |

All three drive the **same** `SpaceGame` object — only the input differs.

## 5. Library roles
- **Pygame** — runs the game loop, input, and rendering (`game/space_game.py`).
- **OpenCV** — opens the webcam, reads/flips/color-converts frames, draws overlays, shows the feed (`camera_play.py`).
- **MediaPipe** — detects the hand; the index-fingertip position becomes a move direction and a closed fist brakes (`vision/hand_tracker.py`).
- **DeepFace** — reads the dominant facial emotion and maps it to game difficulty / pirate speed (`vision/face_analyzer.py`).
- **Gymnasium** — wraps the game as an RL environment with `reset()`/`step()`, `action_space`, `observation_space` (`rl/space_env.py`).
- **Stable-Baselines3** — trains a PPO agent (`train.py`) and runs it (`ai_play.py`).

## 6. State / Actions / Reward / Done
**State** (`observation_space` = `Box(0,1, shape=(6,))`, normalized):
`[player_x, player_y, ore_x, ore_y, nearest_pirate_x, nearest_pirate_y]`

**Actions** (`action_space = Discrete(5)`): `0 left, 1 right, 2 up, 3 down, 4 idle`.

**Reward:** `+1` collect ore · `-1` pirate collision · `-0.01` per step ·
`+0.02 / -0.005` distance-to-ore shaping (toggleable for the report).

**Episode ends when:** pirate collision (fail), target score reached (win),
or step limit exceeded (truncation).

## 7. How the camera affects the game
The webcam frame is read by OpenCV, passed to MediaPipe (hand → movement) and to
DeepFace (face → emotion → difficulty). When **no hand** is detected the ship
idles; when **no face** is detected the last stable difficulty is kept.

## 8. Training process
`train.py` builds a `PPO("MlpPolicy")` over the `SpaceEnv` for `TOTAL_TIMESTEPS`
steps and saves `models/space_model.zip`. `ai_play.py` reloads it and acts with
`deterministic=True`.

## 9. Folder structure
```
space-salvager/
├── manual_play.py  camera_play.py  train.py  ai_play.py
├── game/space_game.py            # pure game logic + rendering
├── vision/hand_tracker.py        # MediaPipe
├── vision/face_analyzer.py       # DeepFace
├── rl/space_env.py               # Gymnasium env
├── models/space_model.zip        # trained agent
├── assets/images/  assets/sounds/
└── docs/project_report.pdf
```

## 10. Install
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 11. Run
```bash
python manual_play.py     # keyboard
python camera_play.py     # webcam: hand + emotion
python train.py           # train the agent (set RENDER/TOTAL_TIMESTEPS inside)
python ai_play.py         # watch the trained agent
```

## 12. Notes for the report (docs/project_report.pdf)
Covers: State/Actions/Reward, Policy, Exploration vs Exploitation, training
algorithm + timesteps, before/after performance, problems encountered, and the
division of work. See the assignment Part יא checklist.

---
_Art and sound assets adapted from the HomeWorld2D project._
