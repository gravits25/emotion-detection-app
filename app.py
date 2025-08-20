import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
from deepface import DeepFace
from utils.anti_spoof import liveness_detection

st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("😊 Real-time Emotion Detection with Anti-Spoofing")
st.write("Detect emotions (Happy, Sad, Angry, Neutral, etc.) from your webcam in real-time.")

# -------------------- Video Transformer --------------------
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = "No face detected"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            # Anti-spoofing check
            if not liveness_detection(img):
                cv2.putText(img, "⚠️ Fake/Static Face Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return img

            # DeepFace analysis
            result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

            # Overlay emotion text
            cv2.putText(img, f"Emotion: {dominant_emotion}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.last_prediction = dominant_emotion

        except Exception as e:
            cv2.putText(img, "Error: No Face", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img


# -------------------- Streamlit WebRTC --------------------
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("👆 Allow webcam access above to start emotion detection.")
