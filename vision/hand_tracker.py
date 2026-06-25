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

    def get_direction(self, frame, deadzone: float = 0.15):
        """Process a BGR frame. Returns ((dx, dy), annotated_frame).

        dx, dy are each in {-1, 0, 1}, derived independently from the index
        fingertip's offset from the frame center, so diagonals are possible
        (e.g. (1, -1) = up-right). A closed fist forces (0, 0) (brake), as does
        no hand detected. `deadzone` is the fraction of the half-frame the
        finger must pass before an axis activates.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        dx = dy = 0  # default when no hand is detected
        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)

            if self._is_fist(lms):
                cv2.putText(frame, "FIST = BRAKE", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                h, w, _ = frame.shape
                cx = lms.landmark[8].x * w
                cy = lms.landmark[8].y * h
                # normalized offset from center in [-1, 1] per axis
                ox = (cx - w / 2) / (w / 2)
                oy = (cy - h / 2) / (h / 2)
                if ox > deadzone:
                    dx = 1
                elif ox < -deadzone:
                    dx = -1
                if oy > deadzone:
                    dy = 1
                elif oy < -deadzone:
                    dy = -1
                cv2.circle(frame, (int(cx), int(cy)), 12, (255, 0, 255), cv2.FILLED)
        else:
            cv2.putText(frame, "No hand detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"Move: ({dx}, {dy})", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return (dx, dy), frame

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
