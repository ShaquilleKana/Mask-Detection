import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import threading
import av
import tempfile
from fractions import Fraction
import base64

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
# SHARED STATS (PERSISTENT)
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

stats = st.session_state.stats

# =====================================================
# VIDEO PROCESSOR (NO CV2)
# =====================================================
class MaskDetectionProcessor:
    def __init__(self, stats):
        self.stats = stats
        self.confidence = 0.5
        self.process_size = (640, 480)

    def recv(self, frame):
        # Frame → numpy
        img = frame.to_ndarray(format="bgr24")

        # Resize pakai PIL (AMAN DEPLOY)
        pil_img = Image.fromarray(img)
        pil_resized = pil_img.resize(self.process_size)
        img_resized = np.array(pil_resized)

        # YOLO inference
        results = model.predict(
            img_resized,
            conf=self.confidence,
            verbose=False
        )

        # Annotated image (YOLO internal)
        annotated = results[0].plot()

        # Resize balik ke ukuran asli
        annotated_pil = Image.fromarray(annotated)
        annotated_pil = annotated_pil.resize((img.shape[1], img.shape[0]))
        annotated = np.array(annotated_pil)

        # Hitung deteksi
        mask_on = 0
        mask_off = 0
        for box in results[0].boxes:
            name = model.names[int(box.cls[0])]
            if name == "Mask_On":
                mask_on += 1
            elif name == "Mask_Off":
                mask_off += 1

        # Update stats (THREAD SAFE)
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
page = st.sidebar.radio("Menu", ["Realtime", "Image", "Video"])
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

# =====================================================
# REALTIME PAGE
# =====================================================
if page == "Realtime":
    st.title("📹 Real-Time Mask Detection")

    # Paksa UI rerun
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
elif page == "Image":
    st.title("🖼️ Image Detection")

    file = st.file_uploader("Upload Image", ["jpg", "jpeg", "png"])
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

# =====================================================
# VIDEO DETECTION
# =====================================================
elif page == "Video":
    st.title("🎥 Video Mask Detection")

    import tempfile
    import base64
    import os
    from fractions import Fraction

    def show_small_video(video_path, width=360):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode()

        st.markdown(
            f"""
            <video width="{width}" controls>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            """,
            unsafe_allow_html=True
        )

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        # ambil nama file asli
        original_name = video_file.name
        name, ext = os.path.splitext(original_name)
        output_filename = f"{name}_detected{ext}"

        # simpan video upload
        input_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        input_file.write(video_file.read())

        st.subheader("📼 Uploaded Video")
        show_small_video(input_file.name, width=360)

        if st.button("Detect Video"):
            st.info("Processing video...")

            container = av.open(input_file.name)
            in_stream = container.streams.video[0]

            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            output_container = av.open(output_file.name, mode="w")

            out_stream = output_container.add_stream("mpeg4", rate=25)
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.time_base = Fraction(1, 25)

            total_on = 0
            total_off = 0
            progress = st.progress(0)

            left, center, right = st.columns([1, 2, 1])
            with center:
                frame_placeholder = st.empty()

            frames = list(container.decode(video=0))
            total_frames = len(frames)

            for i, frame in enumerate(frames):
                img = frame.to_ndarray(format="bgr24")

                results = model.predict(
                    img,
                    conf=confidence,
                    verbose=False
                )

                annotated = results[0].plot()

                for box in results[0].boxes:
                    name_cls = model.names[int(box.cls[0])]
                    if name_cls == "Mask_On":
                        total_on += 1
                    elif name_cls == "Mask_Off":
                        total_off += 1

                frame_placeholder.image(
                    annotated[:, :, ::-1],
                    channels="RGB",
                    width=360
                )

                new_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                packet = out_stream.encode(new_frame)
                if packet:
                    output_container.mux(packet)

                progress.progress((i + 1) / total_frames)

            packet = out_stream.encode(None)
            if packet:
                output_container.mux(packet)

            output_container.close()
            container.close()

            st.success("Video processing finished!")

            st.markdown(
                f"""
                **Output File:** `{output_filename}`  
                **Total Detections:** {total_on + total_off}  
                **Mask On:** {total_on}  
                **Mask Off:** {total_off}
                """
            )

            # 🔽 DOWNLOAD DENGAN NAMA FILE ASLI + _detected
            with open(output_file.name, "rb") as f:
                st.download_button(
                    label="⬇️ Download Result Video",
                    data=f.read(),
                    file_name=output_filename,
                    mime="video/mp4"
                )


