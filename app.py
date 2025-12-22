import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import threading
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Mask Detection", layout="wide")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.fuse()
    return model

model = load_model()

# =====================================================
# SHARED STATS (PERSISTENT !!!)
# =====================================================
class SharedStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.mask_on = 0
        self.mask_off = 0
        self.total = 0
        self.fps = 0.0
        self._frames = 0
        self._last_time = time.time()

if "stats" not in st.session_state:
    st.session_state.stats = SharedStats()

stats = st.session_state.stats   # ⬅️ INI KUNCINYA

# =====================================================
# VIDEO PROCESSOR
# =====================================================
class MaskDetectionProcessor:
    def __init__(self, stats):
        self.stats = stats
        self.confidence = 0.5
        self.process_size = (640, 480)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        pil_img = Image.fromarray(img)
        pil_resized = pil_img.resize(self.process_size)
        img_resized = np.array(pil_resized)

        results = model.predict(
            img_resized,
            conf=self.confidence,
            verbose=False
        )

        annotated = results[0].plot()
        annotated_pil = Image.fromarray(annotated)
        annotated_pil = annotated_pil.resize((img.shape[1], img.shape[0]))
        annotated = np.array(annotated_pil)

        mask_on = 0
        mask_off = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name == "Mask_On":
                mask_on += 1
            elif name == "Mask_Off":
                mask_off += 1

        # 🔒 UPDATE STATS YANG TIDAK RESET
        with self.stats.lock:
            self.stats.mask_on = mask_on
            self.stats.mask_off = mask_off
            self.stats.total = mask_on + mask_off

            self.stats._frames += 1
            if self.stats._frames >= 10:
                now = time.time()
                self.stats.fps = self.stats._frames / (now - self.stats._last_time)
                self.stats._frames = 0
                self.stats._last_time = now

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Settings")
page = st.sidebar.radio("Menu", ["Realtime", "Image"])
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

# =====================================================
# REALTIME PAGE
# =====================================================
if page == "Realtime":
    st.title("📹 Real-Time Mask Detection")

    # 🔥 PAKSA RERUN UI
    st_autorefresh(interval=500, key="refresh")

    col1, col2 = st.columns([2, 1])

    with col1:
        ctx = webrtc_streamer(
            key="mask",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=lambda: MaskDetectionProcessor(stats),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.confidence = confidence

    with col2:
        st.subheader("📊 Statistics")

        with stats.lock:
            st.metric("FPS", f"{stats.fps:.1f}")
            st.metric("Total People", stats.total)
            st.metric("Mask On", stats.mask_on)
            st.metric("Mask Off", stats.mask_off)

# =====================================================
# IMAGE PAGE
# =====================================================
else:
    st.title("🖼️ Image Detection")
    file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Detect"):
            img = np.array(image)[:, :, ::-1]
            results = model.predict(img, conf=confidence)
            annotated = results[0].plot()

            on, off = 0, 0
            for box in results[0].boxes:
                name = model.names[int(box.cls[0])]
                if name == "Mask_On":
                    on += 1
                elif name == "Mask_Off":
                    off += 1

            st.image(annotated[:, :, ::-1], use_container_width=True)
            st.success(f"Total: {on+off} | Mask On: {on} | Mask Off: {off}")
