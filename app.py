import cv2
import streamlit as st
from ultralytics import YOLO
import time

# Set page config
st.set_page_config(page_title="Object Counter", page_icon="📷")

st.title("📷 Water Bottle & Hotpack Counter")
st.write("Detects and counts 'bottle' and 'hotpack' in real-time.")

# Load the model with caching to prevent reloading on every run
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    with st.spinner("Loading Model..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar for controls
st.sidebar.header("Controls")
run = st.sidebar.checkbox('Start Webcam')
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold (NMS)", 0.0, 1.0, 0.5, 0.05, help="Lower value = more aggressive removal of overlapping boxes.")
agnostic_nms = st.sidebar.checkbox("Agnostic NMS", value=True, help="Prevents overlapping detections of different classes.")

# Layout for counts
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 💧 Water Bottles (물병)")
    bottle_count_placeholder = st.empty()
    bottle_count_placeholder.metric("Count", 0)

with col2:
    st.markdown("### 🔥 Hotpacks (핫팩)")
    hotpack_count_placeholder = st.empty()
    hotpack_count_placeholder.metric("Count", 0)

# Placeholder for the video frame
FRAME_WINDOW = st.image([])

if run:
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("Could not open webcam.")
        st.stop()
        
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture image from camera.")
            break
        
        # Run inference using the user-selected parameters
        results = model(frame, conf=confidence_threshold, iou=iou_threshold, agnostic_nms=agnostic_nms)
        
        # Calculate counts
        bottle_count = 0
        hotpack_count = 0
        
        # Check detected boxes
        if results and len(results) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if class_name == 'bottle':
                    bottle_count += 1
                elif class_name == 'hotpack':
                    hotpack_count += 1
        
        # Update metrics
        bottle_count_placeholder.metric("Count", bottle_count)
        hotpack_count_placeholder.metric("Count", hotpack_count)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Convert BGR to RGB for Streamlit/PIL display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        FRAME_WINDOW.image(rgb_frame)
        
        # Small sleep to reduce CPU usage if needed, though streamlit usually handles this
        # time.sleep(0.01) 

    camera.release()
else:
    st.info("Check 'Start Webcam' in the sidebar to begin.")



