"""DeepFace emotion analysis -> game difficulty.

Adapted from useful code/emotion.py. Runs DeepFace only once every N frames
(performance + stabilization) and keeps the last result between runs. Maps the
dominant emotion to a difficulty multiplier that scales pirate speed in the game.

Handles the "no face detected" case by keeping the previous emotion.
"""

from deepface import DeepFace

# emotion -> difficulty multiplier (affects pirate speed in SpaceGame)
EMOTION_DIFFICULTY = {
    "happy": 0.8,      # relaxed -> easier, bonus feel
    "neutral": 1.0,
    "sad": 1.0,
    "fear": 1.2,
    "surprise": 1.3,   # special: speed up
    "angry": 1.4,      # hardest
    "disgust": 1.2,
}


class FaceAnalyzer:
    def __init__(self, every_n_frames: int = 30):
        self.every_n = every_n_frames
        self._count = 0
        self.emotion = "neutral"
        self.difficulty = 1.0

    def analyze(self, frame):
        """Update emotion/difficulty at most once per N frames.

        Returns (emotion, difficulty). On no-face / error, keeps last values.
        """
        self._count += 1
        if self._count % self.every_n != 0:
            return self.emotion, self.difficulty

        try:
            result = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False
            )
            self.emotion = result[0]["dominant_emotion"]
            self.difficulty = EMOTION_DIFFICULTY.get(self.emotion, 1.0)
        except Exception:
            # no face / detector error -> keep previous stable result
            pass
        return self.emotion, self.difficulty
