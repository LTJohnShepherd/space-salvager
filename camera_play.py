"""Camera mode: control the ship with your hand, difficulty set by your emotion.

Pipeline (PDF Part ז):
    OpenCV   -> open camera, read + flip frames, color convert, show window
    MediaPipe-> index-fingertip position becomes a move direction; fist = brake
    DeepFace -> dominant emotion sets game difficulty (pirate speed)
    Pygame   -> the SAME SpaceGame object is driven by those inputs

Press R in the game window to reset, ESC (game) or ESC in the camera window to quit.
Gracefully handles: camera unavailable, no hand detected, no face detected.
"""

import cv2
import pygame
from game.space_game import SpaceGame
from vision.hand_tracker import HandTracker
from vision.face_analyzer import FaceAnalyzer

DIR_TO_ACTION = {
    "left": lambda g: g.move_left(),
    "right": lambda g: g.move_right(),
    "up": lambda g: g.move_up(),
    "down": lambda g: g.move_down(),
    "idle": lambda g: g.idle(),
}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not available. Try manual_play.py instead.")
        return

    game = SpaceGame(render=True, fps=60)
    hands = HandTracker()
    face = FaceAnalyzer(every_n_frames=30)

    running = True
    while running:
        # --- Pygame events (quit / reset) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()

        # --- OpenCV: read + pre-process a frame ---
        ret, frame = cap.read()
        if not ret:
            game.idle()
        else:
            frame = cv2.flip(frame, 1)

            # --- MediaPipe: hand -> direction ---
            direction, frame = hands.get_direction(frame)
            DIR_TO_ACTION.get(direction, DIR_TO_ACTION["idle"])(game)

            # --- DeepFace: emotion -> difficulty ---
            emotion, difficulty = face.analyze(frame)
            game.difficulty = difficulty
            game.overlay_lines = [
                "Camera Mode",
                f"Hand: {direction}",
                f"Emotion: {emotion}",
            ]
            cv2.putText(frame, f"Emotion: {emotion}", (20, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Camera (ESC to quit)", frame)

        # --- advance + draw the game ---
        if not game.is_done():
            game.update()
        game.render()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC in camera window
            running = False

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    game.close()


if __name__ == "__main__":
    main()
