import streamlit as st
import cv2
import time
import tempfile
import os
from datetime import datetime
import pandas as pd
import numpy as np
from ultralytics import YOLO
import base64

# Page configuration
st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitTrafficDetector:
    def __init__(self, model_path='yolov8s.pt'):
        """Initialize the traffic detector for Streamlit"""
        if 'model' not in st.session_state:
            with st.spinner("Loading YOLOv8 model..."):
                st.session_state.model = YOLO(model_path)
        
        self.model = st.session_state.model
        
        # Define classes we're interested in
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.person_class = [0]  # person
        
        # Class names for display
        self.class_names = {
            0: 'person', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck'
        }
        
        # Colors for different classes (BGR format)
        self.colors = {
            0: (0, 255, 0),    # person - green
            2: (255, 0, 0),    # car - blue
            3: (0, 0, 255),    # motorcycle - red
            5: (255, 255, 0),  # bus - cyan
            7: (255, 0, 255)   # truck - magenta
        }
        
        # Initialize detection log
        if 'detection_log' not in st.session_state:
            st.session_state.detection_log = []
    
    def detect_frame(self, frame, frame_number, timestamp):
        """Detect objects in a single frame and log detections"""
        results = self.model(frame, verbose=False)
        
        # Log detections
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()
                
                # Only log vehicles and people with high confidence
                if class_id in self.vehicle_classes + self.person_class and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    detection_entry = {
                        'timestamp': timestamp,
                        'frame': frame_number,
                        'object_type': self.class_names.get(class_id, 'unknown'),
                        'confidence': round(confidence, 3),
                        'bbox': f"({x1},{y1})-({x2},{y2})",
                        'center_x': int((x1 + x2) / 2),
                        'center_y': int((y1 + y2) / 2)
                    }
                    st.session_state.detection_log.append(detection_entry)
        
        return results[0]
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on the frame"""
        boxes = results.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Only draw boxes for vehicles and people
                if class_id in self.vehicle_classes + self.person_class:
                    # Get color for this class
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{self.class_names.get(class_id, 'unknown')}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, input_path, progress_bar, status_text):
        """Process video and return path to processed video"""
        # Create temporary output file
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            st.error(f"Error: Could not open video {input_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Clear previous detection log
        st.session_state.detection_log = []
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_count / fps
            
            # Detect objects
            results = self.detect_frame(frame, frame_count, timestamp)
            
            # Draw detections
            processed_frame = self.draw_detections(frame.copy(), results)
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add FPS counter to frame
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            writer.write(processed_frame)
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
        
        # Cleanup
        cap.release()
        writer.release()
        
        return output_path

def get_video_download_link(video_path, filename="processed_video.mp4"):
    """Generate download link for processed video"""
    with open(video_path, "rb") as file:
        video_bytes = file.read()
    
    b64_video = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64_video}" download="{filename}">Download Processed Video</a>'
    return href

def display_detection_stats():
    """Display detection statistics"""
    if st.session_state.detection_log:
        df = pd.DataFrame(st.session_state.detection_log)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(df))
        
        with col2:
            st.metric("Unique Objects", df['object_type'].nunique())
        
        with col3:
            most_common = df['object_type'].mode().iloc[0] if not df.empty else "None"
            st.metric("Most Detected", most_common)
        
        with col4:
            avg_confidence = df['confidence'].mean() if not df.empty else 0
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Object type distribution
        st.subheader("Detection Summary")
        object_counts = df['object_type'].value_counts()
        st.bar_chart(object_counts)
        
        return df
    else:
        st.info("No detections logged yet. Process a video to see detection statistics.")
        return None

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🚦 Traffic Violation Detection System")
    st.markdown("Upload a traffic video to detect vehicles and pedestrians using YOLOv8")
    
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.markdown("### Model Configuration")
    
    # Model selection
    model_options = {
        "YOLOv8n (Fast)": "yolov8n.pt",
        "YOLOv8s (Balanced)": "yolov8s.pt", 
        "YOLOv8m (Accurate)": "yolov8m.pt"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select YOLO Model",
        options=list(model_options.keys()),
        index=1  # Default to YOLOv8s
    )
    
    # Initialize detector
    detector = StreamlitTrafficDetector(model_options[selected_model])
    
    # File uploader
    st.header("📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a traffic video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file containing traffic footage"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_input_path = tmp_file.name
        
        # Display video info
        cap = cv2.VideoCapture(temp_input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        st.success(f"✅ Video uploaded successfully!")
        st.info(f"📊 Video Info: {width}x{height} @ {fps}fps, Duration: {duration:.1f}s, Frames: {total_frames}")
        
        # Process button
        if st.button("🚀 Process Video", type="primary"):
            
            with st.spinner("Processing video... This may take a few minutes."):
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                processed_video_path = detector.process_video(
                    temp_input_path, 
                    progress_bar, 
                    status_text
                )
                
                if processed_video_path and os.path.exists(processed_video_path):
                    st.success("✅ Video processing completed!")
                    
                    # Display videos side by side
                    st.header("📺 Video Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Video")
                        st.video(temp_input_path)
                    
                    with col2:
                        st.subheader("Processed Video")
                        st.video(processed_video_path)
                    
                    # Download button
                    st.header("💾 Download Results")
                    
                    # Create download link
                    download_link = get_video_download_link(
                        processed_video_path, 
                        f"processed_{uploaded_file.name}"
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    # Detection statistics
                    st.header("📊 Detection Results")
                    detection_df = display_detection_stats()
                    
                    if detection_df is not None:
                        # Show detailed log
                        with st.expander("📋 Detailed Detection Log"):
                            st.dataframe(
                                detection_df, 
                                use_container_width=True,
                                height=300
                            )
                        
                        # Download detection log
                        csv = detection_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Detection Log (CSV)",
                            data=csv,
                            file_name=f"detections_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                    
                    # Cleanup processed video file
                    try:
                        os.unlink(processed_video_path)
                    except:
                        pass
                
                else:
                    st.error("❌ Error processing video. Please try again.")
        
        # Cleanup temporary input file
        try:
            os.unlink(temp_input_path)
        except:
            pass
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a traffic video file to get started")
        
        # Show example or demo section
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Upload** a traffic video file (MP4, AVI, MOV, MKV)
        2. **Select** your preferred YOLO model (faster vs more accurate)
        3. **Process** the video to detect vehicles and pedestrians
        4. **View** original and processed videos side-by-side
        5. **Download** the processed video and detection logs
        
        ### Detected Objects:
        - 🚗 **Cars** (Blue boxes)
        - 🏍️ **Motorcycles** (Red boxes)  
        - 🚌 **Buses** (Cyan boxes)
        - 🚛 **Trucks** (Magenta boxes)
        - 🚶 **People** (Green boxes)
        """)

if __name__ == "__main__":
    main()
