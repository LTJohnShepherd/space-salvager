"""Generates docs/project_report.pdf. Run once: python docs/make_report.py
(reportlab is only needed for this generator, not for the game itself.)"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, ListFlowable, ListItem)

OUT = os.path.join(os.path.dirname(__file__), "project_report.pdf")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=10,
                    spaceAfter=4, textColor=colors.HexColor("#13355b"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8,
                    spaceAfter=2, textColor=colors.HexColor("#1f5fa6"))
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9.6, leading=13,
                      spaceAfter=4, alignment=4)
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontSize=22,
                       textColor=colors.HexColor("#0e2a4a"))
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=10.5,
                     textColor=colors.HexColor("#555555"), spaceAfter=2)

S = []
def P(t, st=BODY): S.append(Paragraph(t, st))
def bullets(items):
    S.append(ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=10) for i in items],
                          bulletType="bullet", start="circle", leftIndent=12))

# ---------------- Title ----------------
P("Space Salvager", TITLE)
P("Final Project - AI-Based Game Development", SUB)
P("Python | Pygame | OpenCV | MediaPipe | DeepFace | "
  "Gymnasium | Stable-Baselines3", SUB)
P("<b>Submitter:</b> [your name + ID]      "
  "<b>Repository:</b> github.com/LTJohnShepherd/space-salvager", SUB)
S.append(Spacer(1, 6))

# ---------------- 1. Game ----------------
P("1. Game Description", H1)
P("<b>Space Salvager</b> is a 2D arcade game built with Pygame. The player pilots a single "
  "salvager ship in an asteroid field. Flying into an ore pickup scores a point and spawns "
  "new ore; touching any of the roaming pirate cruisers destroys the ship and ends the run. "
  "The goal is to collect a target number of ore (15) without colliding with a pirate.")
P("<b>Rules &amp; ending.</b> Each ore = +1 score. A pirate collision ends the episode "
  "(failure). The episode also ends on reaching the target score (win) or exceeding the step "
  "limit. The same game object is driven three ways: keyboard (manual), webcam hand gesture + "
  "facial emotion (camera), and a trained reinforcement-learning agent (AI).")

# ---------------- 2. Libraries ----------------
P("2. Role of Each Library", H1)
bullets([
 "<b>Pygame</b> - runs the game loop, input, rendering and sound (game/space_game.py, manual_play.py).",
 "<b>OpenCV</b> - opens the webcam (VideoCapture), reads and flips frames, converts BGR to RGB, draws overlays, and shows the feed (camera_play.py).",
 "<b>MediaPipe</b> - MediaPipe Hands detects the hand; the index-fingertip landmark (id 8) is converted into a movement direction, and a closed fist is detected as a brake (vision/hand_tracker.py).",
 "<b>DeepFace</b> - analyses the dominant facial emotion and maps it to a difficulty multiplier that scales pirate speed (vision/face_analyzer.py).",
 "<b>Gymnasium</b> - wraps the game as an RL environment with reset()/step() and defined action and observation spaces (rl/space_env.py).",
 "<b>Stable-Baselines3</b> - trains a PPO agent (train.py) and runs it autonomously (ai_play.py).",
])

# ---------------- 3. Camera pipeline ----------------
P("3. Connecting the Camera to the Game", H1)
P("OpenCV captures each frame and flips it horizontally so movement feels like a mirror. The "
  "frame is passed to MediaPipe, which returns hand landmarks; the index fingertip's offset "
  "from the frame centre is thresholded per-axis (a dead-zone) into a direction vector "
  "(dx, dy), each in {-1, 0, 1}, allowing the eight compass directions plus idle. The same "
  "frame is passed to DeepFace, whose dominant emotion sets the difficulty multiplier. "
  "<b>No-detection handling:</b> if no hand is found the ship idles; if no face is found the "
  "last stable difficulty is kept. DeepFace runs only once every N frames and reuses its last "
  "result, both for performance and to avoid flicker.")

# ---------------- 4. State ----------------
P("4. State / Observation", H1)
P("The observation is a vector of <b>6 floats, normalized to [0,1]</b>: "
  "<font face='Courier'>[player_x, player_y, ore_x, ore_y, nearest_pirate_x, nearest_pirate_y]</font>. "
  "These are the minimum values needed to decide where to go: where I am, where the reward is, "
  "and where the closest threat is. Normalization keeps the neural-network inputs in a stable "
  "range. <b>Sufficiency / limitation:</b> the agent sees only the nearest pirate (not all "
  "three) and has no velocity information, so it cannot perfectly anticipate pirate paths - "
  "a deliberate simplification that keeps the state small and learnable, and a clear candidate "
  "for future extension.")

# ---------------- 5. Actions ----------------
P("5. Actions", H1)
P("The action space is <b>Discrete(9)</b>: idle, the four straight moves, and the four "
  "diagonals. The environment maps each integer to a direction vector (the ACTION_VECTORS "
  "table) and calls set_velocity(dx, dy); the vector is normalized so diagonal moves are not "
  "faster than straight ones. Nine discrete actions give full 2D mobility while keeping the "
  "action space discrete, which suits PPO and DQN. (An earlier version used Discrete(5) with no "
  "diagonals - comparing the two is a useful experiment.)")

# ---------------- 6. Reward ----------------
P("6. Reward", H1)
P("The reward signal is: <b>+1</b> for collecting ore, <b>-1</b> for a pirate collision "
  "(terminal), <b>-0.01</b> per step to encourage efficiency, plus a small distance-shaping "
  "term (<b>+0.02</b> when the ship moves closer to the ore, <b>-0.005</b> when it moves away). "
  "This encourages the agent to reach ore quickly while avoiding pirates. <b>Reward shaping "
  "during development:</b> the first version used only the sparse +1/-1/-0.01 signal and learned "
  "slowly; adding the distance term gave the agent a denser gradient toward the ore and sped up "
  "learning - the two variants are compared in the Results section.")

# ---------------- 7. Policy ----------------
P("7. Policy", H1)
P("The policy is the function (a neural network, MlpPolicy) that maps a state to a distribution "
  "over the nine actions. During training the agent samples from this distribution; at play time "
  "it takes the most likely action (deterministic). PPO updates the policy's weights via policy "
  "gradients so that actions which led to higher reward become more probable. The policy depends "
  "entirely on the 6-value state described above.")

# ---------------- 8. Exploration vs Exploitation ----------------
P("8. Exploration vs. Exploitation", H1)
P("<b>Exploration</b> means trying varied actions to discover which lead to reward; "
  "<b>exploitation</b> means choosing the actions already known to be good. During training PPO "
  "explores by sampling stochastically from the policy (encouraged by an entropy term). At play "
  "time we call <font face='Courier'>predict(obs, deterministic=True)</font>, which is pure "
  "exploitation - always the highest-probability action. Exploration matters most early on: "
  "before the agent has ever reached ore or been hit, it must try many actions to gather the "
  "experience the policy learns from.")

# ---------------- 9. Training ----------------
P("9. Training Process", H1)
P("The agent is trained with <b>PPO (MlpPolicy)</b> from Stable-Baselines3. PPO was chosen as a "
  "stable, well-documented on-policy algorithm that works well with discrete actions and small "
  "observation vectors. Training runs headless for speed for <b>[TOTAL_TIMESTEPS, e.g. 500,000]</b> "
  "timesteps and saves models/space_model.zip; ai_play.py reloads it and acts deterministically. "
  "At the start of training the agent moved almost randomly and died quickly; as training "
  "progressed it began steering toward ore and surviving longer. The action space was changed "
  "from 5 to 9 actions mid-project, which required retraining a fresh model.")

# ---------------- 10. Results ----------------
P("10. Results", H1)
P("Performance was measured with the mean episode reward (<font face='Courier'>ep_rew_mean</font>) "
  "reported by Stable-Baselines3, comparing the start of training to the end. Replace the bracketed "
  "values below with the numbers from your own training log.")
tbl = Table([
    ["Metric", "Before training", "After training"],
    ["Mean episode reward", "[~ -1.0]", "[your value]"],
    ["Mean episode length (steps)", "[short]", "[longer]"],
    ["Ore collected per episode", "[~0]", "[your value]"],
], colWidths=[7*cm, 4*cm, 4*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f5fa6")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#aaaaaa")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3fa")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
S.append(tbl)
S.append(Spacer(1, 4))
P("Two reward variants were compared: the sparse signal (+1/-1/-0.01) and the same signal plus "
  "distance shaping. The shaped variant reached a positive mean reward in fewer timesteps.")

# ---------------- 11. Problems ----------------
P("11. Problems Encountered and How They Were Solved", H1)
bullets([
 "<b>Python 3.14 had no wheels</b> for pygame/mediapipe/stable-baselines3 - solved by using a Python 3.12 virtual environment.",
 "<b>pip and python pointed to different interpreters</b> (packages installed but not found) - solved by installing with python -m pip inside the venv.",
 "<b>Sprites looked squished</b> because images were scaled to a square - solved by scaling the longest edge while preserving the native aspect ratio.",
 "<b>The pirate sprite looked too small</b> - solved by decoupling on-screen display size from the collision radius.",
 "<b>Audio played too often</b> (a sound on every episode reset) - solved by playing the launch sound once, lowering the master volume, and adding an M mute toggle.",
 "<b>The trained model became incompatible</b> after changing the action space - solved by deleting and retraining the model.",
])

# ---------------- 12. Conclusions ----------------
P("12. Conclusions", H1)
P("PPO successfully learned to navigate toward ore and avoid the nearest pirate from a compact "
  "6-value state. The hardest parts were the environment setup (Python version and venv issues) "
  "and tuning the reward so the agent learned efficiently rather than wandering. Future "
  "improvements: include pirate velocities and all pirates in the state, compare PPO with DQN, "
  "experiment further with reward shaping, and add levels or moving ore.")

# ---------------- 13. Division of work ----------------
P("13. Division of Work", H1)
P("[Describe how the work was divided among team members, or state that this was an individual "
  "project. Example: \"Individual project - all components (game, vision, RL environment, "
  "training, documentation) implemented by the submitter.\"]")

SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.4*cm, bottomMargin=1.4*cm,
                  leftMargin=1.6*cm, rightMargin=1.6*cm,
                  title="Space Salvager - Project Report").build(S)
print("wrote", OUT)
