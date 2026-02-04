import math
import time
from typing import Dict, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


st.set_page_config(page_title="Hand Sign Detector", layout="centered")

st.title("Hand Sign Detector")
st.write(
    "Simple hand sign recognition using MediaPipe Hands. "
    "Signs: Thumbs Up, Thumbs Down, Peace, OK, Fist, Open Palm."
)


mp_hands = mp.solutions.hands


FINGER_TIPS = {
    "thumb": mp_hands.HandLandmark.THUMB_TIP,
    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp_hands.HandLandmark.PINKY_TIP,
}

FINGER_PIPS = {
    "index": mp_hands.HandLandmark.INDEX_FINGER_PIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_PIP,
    "pinky": mp_hands.HandLandmark.PINKY_PIP,
}

THUMB_IP = mp_hands.HandLandmark.THUMB_IP


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lm_xy(landmarks, idx) -> Tuple[float, float]:
    lm = landmarks[idx]
    return lm.x, lm.y


def _finger_extended(landmarks, tip, pip, wrist, threshold=0.02) -> bool:
    tip_xy = _lm_xy(landmarks, tip)
    pip_xy = _lm_xy(landmarks, pip)
    wrist_xy = _lm_xy(landmarks, wrist)
    return _dist(tip_xy, wrist_xy) > _dist(pip_xy, wrist_xy) + threshold


def _thumb_extended(landmarks, threshold=0.02) -> bool:
    tip_xy = _lm_xy(landmarks, FINGER_TIPS["thumb"])
    ip_xy = _lm_xy(landmarks, THUMB_IP)
    wrist_xy = _lm_xy(landmarks, mp_hands.HandLandmark.WRIST)
    return _dist(tip_xy, wrist_xy) > _dist(ip_xy, wrist_xy) + threshold


def _finger_states(landmarks) -> Dict[str, bool]:
    wrist = mp_hands.HandLandmark.WRIST
    states = {
        "thumb": _thumb_extended(landmarks),
        "index": _finger_extended(landmarks, FINGER_TIPS["index"], FINGER_PIPS["index"], wrist),
        "middle": _finger_extended(landmarks, FINGER_TIPS["middle"], FINGER_PIPS["middle"], wrist),
        "ring": _finger_extended(landmarks, FINGER_TIPS["ring"], FINGER_PIPS["ring"], wrist),
        "pinky": _finger_extended(landmarks, FINGER_TIPS["pinky"], FINGER_PIPS["pinky"], wrist),
    }
    return states


def _is_thumb_up(landmarks, states: Dict[str, bool]) -> bool:
    tip_y = landmarks[FINGER_TIPS["thumb"]].y
    ip_y = landmarks[THUMB_IP].y
    mcp_y = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    other_curled = not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]
    return states["thumb"] and other_curled and (tip_y < ip_y < mcp_y)


def _is_thumb_down(landmarks, states: Dict[str, bool]) -> bool:
    tip_y = landmarks[FINGER_TIPS["thumb"]].y
    ip_y = landmarks[THUMB_IP].y
    mcp_y = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    other_curled = not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]
    return states["thumb"] and other_curled and (tip_y > ip_y > mcp_y)


def _is_peace(states: Dict[str, bool]) -> bool:
    return states["index"] and states["middle"] and not states["ring"] and not states["pinky"]


def _is_fist(states: Dict[str, bool]) -> bool:
    return not any(states.values())


def _is_open_palm(states: Dict[str, bool]) -> bool:
    return all(states.values())


def _is_ok(landmarks, states: Dict[str, bool]) -> bool:
    thumb_tip = _lm_xy(landmarks, FINGER_TIPS["thumb"])
    index_tip = _lm_xy(landmarks, FINGER_TIPS["index"])
    close = _dist(thumb_tip, index_tip) < 0.05
    other_extended = states["middle"] and states["ring"] and states["pinky"]
    return close and other_extended


def classify_sign(landmarks) -> Optional[str]:
    states = _finger_states(landmarks)
    if _is_thumb_up(landmarks, states):
        return "Thumbs Up"
    if _is_thumb_down(landmarks, states):
        return "Thumbs Down"
    if _is_ok(landmarks, states):
        return "OK"
    if _is_peace(states):
        return "Peace"
    if _is_open_palm(states):
        return "Open Palm"
    if _is_fist(states):
        return "Fist"
    return None


class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=2,
        )
        self.drawer = mp.solutions.drawing_utils
        self.last_states = []
        self.last_labels = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        self.last_states = []
        self.last_labels = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                self.drawer.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    self.drawer.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.drawer.DrawingSpec(color=(0, 0, 255), thickness=2),
                )
                label = classify_sign(hand_landmarks.landmark)
                states = _finger_states(hand_landmarks.landmark)
                self.last_labels.append(label)
                self.last_states.append(states)

        if self.last_labels:
            for idx, label in enumerate(self.last_labels):
                if not label:
                    continue
                y = 30 + idx * 30
                cv2.putText(
                    img,
                    f"Hand {idx + 1}: {label}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.subheader("Camera")
webrtc_ctx = webrtc_streamer(
    key="hand-signs",
    video_processor_factory=HandSignProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

show_states = st.checkbox("Show finger states", value=True)
auto_refresh = st.checkbox("Auto-refresh states", value=True)
states_box = st.empty()

if show_states:
    if webrtc_ctx.video_processor and webrtc_ctx.video_processor.last_states:
        payload = []
        for idx, states in enumerate(webrtc_ctx.video_processor.last_states):
            payload.append(
                {
                    "hand": idx + 1,
                    "sign": webrtc_ctx.video_processor.last_labels[idx],
                    "fingers": states,
                }
            )
        states_box.json(payload)
    else:
        states_box.info("No hand detected.")

st.caption(
    "Tips: Keep your hand in frame, palm facing the camera. "
    "If recognition is unstable, pause and re-center."
)

if auto_refresh and webrtc_ctx.state.playing:
    time.sleep(0.1)
    st.experimental_rerun()
