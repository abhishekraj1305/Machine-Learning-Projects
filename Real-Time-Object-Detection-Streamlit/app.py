import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO

st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("Live Webcam Object Detection (YOLO + Streamlit)")

@st.cache_resource
def load_model(model_name: str):
    return YOLO(model_name)

model_name = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
conf = st.sidebar.slider("Confidence", 0.05, 0.90, 0.35, 0.05)
iou = st.sidebar.slider("IoU", 0.10, 0.90, 0.45, 0.05)
imgsz = st.sidebar.selectbox("Image size", [320, 480, 640, 960], index=2)

model = load_model(model_name)

class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )
        annotated = results[0].plot()  # draws boxes
        return annotated

st.info("Click **Start** and allow webcam permission in browser.")
webrtc_streamer(
    key="yolo-live",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
