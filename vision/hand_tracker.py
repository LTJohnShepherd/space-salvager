"""MediaPipe hand tracking -> game movement direction.

Adapted from useful code/app.py. Returns a discrete direction derived from
the index-fingertip position, plus a 'fist' brake gesture. Also draws the
landmarks/overlay onto the frame so camera_play can show feedback.

Handles the "no hand detected" case by returning direction "idle".
"""

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _is_fist(self, lms):
        """Rough fist check: fingertips below their PIP joints (curled)."""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        curled = sum(1 for t, p in zip(tips, pips)
                     if lms.landmark[t].y > lms.landmark[p].y)
        return curled >= 3

    def get_direction(self, frame):
        """Process a BGR frame. Returns (direction, annotated_frame).

        direction in {"left","right","up","down","idle"}.
        Index fingertip position relative to frame center picks the dominant
        axis; a closed fist forces "idle" (brake).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        direction = "idle"  # default when no hand is detected
        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

            if self._is_fist(lms):
                direction = "idle"
                cv2.putText(frame, "FIST = BRAKE", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                h, w, _ = frame.shape
                cx = lms.landmark[8].x * w
                cy = lms.landmark[8].y * h
                dx = cx - w / 2
                dy = cy - h / 2
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
                cv2.circle(frame, (int(cx), int(cy)), 12, (255, 0, 255), cv2.FILLED)
        else:
            cv2.putText(frame, "No hand detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"Move: {direction}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return direction, frame

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
