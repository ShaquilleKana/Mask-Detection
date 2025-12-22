import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime
import base64

# Page config with dark theme
st.set_page_config(
    page_title="Mask Detection System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #262730;
        color: white;
    }
    .stat-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
        color: #00ff00;
    }
    .stat-label {
        font-size: 1.2em;
        color: #ffffff;
    }
    .warning-text {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Function to play alert sound (browser-based)
def play_alert_sound():
    audio_html = """
    <audio autoplay>
        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUKrk8bllHAU2jdXyzn0vBSp+zPLaizsKGGS57OihUxELTKXh8L1rIQUrlc/y24k3CBxqvvDlm04MD1Cr5PK6ZRsFOI/W8tGAMQUrgs/y24o4CBtpvO/jm04NEFKs5fK8aB0GOpHX8tGBMgUug9Dy24o4CBxqvO/kmk4NEVOt5fK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgUug9Hy3Is5CBxpvO/kmk4NEVOt5vK9aR4GOpHY8tKBMgU=" type="audio/wav">
    </audio>
    """
    return audio_html

# Function to convert image for download
def get_image_download_link(img, filename):
    buffered = io.BytesIO()
    img_pil = Image.fromarray(img)
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">Download Result Image</a>'
    return href

# Sidebar
st.sidebar.title("⚙️ Settings")
page = st.sidebar.radio("Navigation", ["📹 Real-Time Detection", "🖼️ Image Detection"])
st.sidebar.markdown("---")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
enable_sound = st.sidebar.checkbox("🔊 Enable Sound Alert", value=True)

# Initialize session state
if 'stop_camera' not in st.session_state:
    st.session_state.stop_camera = False

# PAGE 1: Real-Time Detection
if page == "📹 Real-Time Detection":
    st.title("📹 Real-Time Mask Detection")
    st.markdown("Monitor mask compliance in real-time using your webcam")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Live Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("### Detection Statistics")
        stats_placeholder = st.empty()
        alert_placeholder = st.empty()
    
    # Camera controls
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_button = st.button("▶️ Start Camera", use_container_width=True)
    with col_btn2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    
    if stop_button:
        st.session_state.stop_camera = True
    
    if start_button:
        st.session_state.stop_camera = False
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Camera cannot be accessed")
        else:
            total_detections = 0
            mask_on_count = 0
            mask_off_count = 0
            
            while not st.session_state.stop_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read frame")
                    break
                
                # YOLO inference
                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()
                
                # Count detections
                boxes = results[0].boxes
                current_mask_off = 0
                current_mask_on = 0
                
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    if class_name == "Mask_Off":
                        current_mask_off += 1
                    elif class_name == "Mask_On":
                        current_mask_on += 1
                
                mask_on_count = current_mask_on
                mask_off_count = current_mask_off
                total_detections = mask_on_count + mask_off_count
                
                # Convert BGR → RGB
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(
                    annotated_frame,
                    channels="RGB",
                    use_container_width=True
                )
                
                # Display statistics
                with stats_placeholder.container():
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number">{total_detections}</div>
                        <div class="stat-label">Total People</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number" style="color: #00ff00;">{mask_on_count}</div>
                            <div class="stat-label">Mask On ✅</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_stat2:
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number" style="color: #ff4444;">{mask_off_count}</div>
                            <div class="stat-label">Mask Off ❌</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if total_detections > 0:
                        compliance = (mask_on_count / total_detections) * 100
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number" style="color: #00d4ff;">{compliance:.1f}%</div>
                            <div class="stat-label">Compliance Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Alert if mask off detected
                if mask_off_count > 0 and enable_sound:
                    with alert_placeholder:
                        st.markdown('<p class="warning-text">⚠️ MASK VIOLATION DETECTED!</p>', unsafe_allow_html=True)
                        st.markdown(play_alert_sound(), unsafe_allow_html=True)
                else:
                    alert_placeholder.empty()
                
                # Small delay to reduce CPU usage
                time.sleep(0.03)
            
            cap.release()
            st.success("✅ Camera stopped successfully")

# PAGE 2: Image Detection
elif page == "🖼️ Image Detection":
    st.title("🖼️ Image Mask Detection")
    st.markdown("Upload an image to detect mask compliance")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Process button
        if st.button("🔍 Detect Masks", use_container_width=True):
            with st.spinner("Processing..."):
                # YOLO inference
                results = model(image, conf=confidence)
                annotated_image = results[0].plot()
                
                # Convert BGR → RGB
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                # Count detections
                boxes = results[0].boxes
                mask_on_count = 0
                mask_off_count = 0
                detections_info = []
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    if class_name == "Mask_Off":
                        mask_off_count += 1
                    elif class_name == "Mask_On":
                        mask_on_count += 1
                    
                    detections_info.append({
                        "Class": class_name,
                        "Confidence": f"{conf:.2%}"
                    })
                
                total_detections = mask_on_count + mask_off_count
                
                with col2:
                    st.markdown("### Detection Result")
                    st.image(annotated_image_rgb, use_container_width=True)
                
                # Statistics
                st.markdown("---")
                st.markdown("### 📊 Detection Statistics")
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number">{total_detections}</div>
                        <div class="stat-label">Total People</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col2:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" style="color: #00ff00;">{mask_on_count}</div>
                        <div class="stat-label">Mask On ✅</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col3:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" style="color: #ff4444;">{mask_off_count}</div>
                        <div class="stat-label">Mask Off ❌</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col4:
                    if total_detections > 0:
                        compliance = (mask_on_count / total_detections) * 100
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-number" style="color: #00d4ff;">{compliance:.1f}%</div>
                            <div class="stat-label">Compliance</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detection details
                if detections_info:
                    st.markdown("---")
                    st.markdown("### 🔍 Detection Details")
                    
                    # Create table
                    import pandas as pd
                    df = pd.DataFrame(detections_info)
                    df.index = df.index + 1
                    df.index.name = "Person #"
                    st.dataframe(df, use_container_width=True)
                
                # Download button
                st.markdown("---")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mask_detection_{timestamp}.jpg"
                
                # Convert for download
                buffered = io.BytesIO()
                img_pil = Image.fromarray(annotated_image_rgb)
                img_pil.save(buffered, format="JPEG")
                
                st.download_button(
                    label="⬇️ Download Result",
                    data=buffered.getvalue(),
                    file_name=filename,
                    mime="image/jpeg",
                    use_container_width=True
                )
                
                # Alert if violations detected
                if mask_off_count > 0:
                    st.error(f"⚠️ WARNING: {mask_off_count} person(s) detected without mask!")
                    if enable_sound:
                        st.markdown(play_alert_sound(), unsafe_allow_html=True)
                else:
                    st.success("✅ All detected persons are wearing masks!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This application uses YOLOv8 for real-time mask detection.
- **Mask_On**: Person wearing mask ✅
- **Mask_Off**: Person not wearing mask ❌
""")