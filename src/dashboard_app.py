import streamlit as st
import os
import sys
import glob
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime
import base64  # Add base64 for image encoding/decoding
import numpy as np
import streamlit.components.v1 as components  # For JavaScript components

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dashboard_utils import *
from src.model import EmotionRecognitionModel
from src.questionnaire import load_questionnaire


# --- Page Configuration ---
st.set_page_config(
    page_title="MindSight - Clinical Assessment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "MindSight - Facial Emotion Recognition for Mental Health Assessment"
    }
)

# --- Custom CSS for Medical UI ---
st.markdown("""
    <style>
    /* Overall styling */
    .main {
        background-color: #F8F9FA;
        padding: 1.5rem;
    }
    .stApp {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0D47A1;
        font-weight: 600;
    }
    h1 {
        border-bottom: 2px solid #0D47A1;
        padding-bottom: 0.5rem;
        font-size: 2.2rem !important;
    }
    h2 {
        font-size: 1.8rem !important;
        margin-top: 1.5rem;
    }
    h3 {
        font-size: 1.5rem !important;
        margin-top: 1rem;
    }
    
    /* Containers */
    .css-1lcbmhc, .css-18e3th9 {
        padding: 1rem 2rem;
    }
    
    /* Cards */
    .card {
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .info-minimal {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .info-mild {
        background-color: #FFF3E0;
        border-left: 5px solid #FFC107;
    }
    .info-moderate {
        background-color: #FBE9E7;
        border-left: 5px solid #FF9800;
    }
    .info-high {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    
    /* Clinical notes area */
    .clinical-notes {
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    
    /* Logo and header */
    .logo-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .logo-text {
        color: #0D47A1;
        font-size: 2rem;
        font-weight: bold;
        margin-left: 10px;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1976D2;
        color: white;
        border-radius: 20px;
        padding: 2px 15px;
        font-weight: 500;
    }
    div.stButton > button:hover {
        background-color: #0D47A1;
        border-color: #0D47A1;
    }
    
    /* Report styling */
    .report-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #0D47A1;
    }
    .report-section {
        margin-top: 1rem;
    }
    .report-item {
        padding: 0.5rem 0;
        border-bottom: 1px dashed #E0E0E0;
    }
    
    /* Metrics */
    div.stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #E3F2FD;
    }
    </style>
    
    <!-- Add JavaScript to request camera permission on page load -->
    <script>
    // Function to request camera permission
    function requestCameraPermission() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                console.log("Camera permission granted");
                // Stop the stream since we just wanted permission
                stream.getTracks().forEach(track => track.stop());
            })
            .catch(function(err) {
                console.error("Error accessing camera: ", err);
            });
    }
    
    // Request permission when the page loads
    window.addEventListener('load', function() {
        // Small delay to ensure the page is fully loaded
        setTimeout(requestCameraPermission, 1000);
    });
    </script>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'welcome'
if 'emotion_history' not in st.session_state:
    st.session_state['emotion_history'] = []
if 'questionnaire_results' not in st.session_state:
    st.session_state['questionnaire_results'] = None
if 'patient_info' not in st.session_state:
    st.session_state['patient_info'] = {
        'id': '',
        'name': '',
        'age': '',
        'gender': '',
        'notes': ''
    }
if 'assessment_complete' not in st.session_state:
    st.session_state['assessment_complete'] = False
if 'combined_assessment' not in st.session_state:
    st.session_state['combined_assessment'] = None
if 'webcam_active' not in st.session_state:
    st.session_state['webcam_active'] = False

# Always use OpenCV for webcam (removed JavaScript webcam implementation)
USE_JS_WEBCAM = False

# --- Header with Logo ---
def render_header():
    st.markdown("""
        <div class="logo-header">
            <div class="logo-text">MindSight 🧠</div>
        </div>
        <p style="margin-top:-15px;color:#616161;font-style:italic;">
            Clinical Mental Health Assessment System
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- Model Loading ---
@st.cache_resource
def load_fer_model(model_path):
    """Load the Facial Emotion Recognition model"""
    # Check for available hardware acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"✓ Using GPU: {device_name}")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device = torch.device("mps")
        st.sidebar.success("✓ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        st.sidebar.info("ⓘ Using CPU for inference (slower)")
    
    # Load model onto selected device
    model = EmotionRecognitionModel(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

# --- Sidebar Navigation ---
def render_sidebar():
    """Render the sidebar navigation and information"""
    with st.sidebar:
        st.markdown("<h3>MindSight Navigation</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Show patient info if available
        if st.session_state['patient_info']['name']:
            st.markdown("<h3>Patient Information</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>ID</strong>: {st.session_state['patient_info']['id']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Name</strong>: {st.session_state['patient_info']['name']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Age</strong>: {st.session_state['patient_info']['age']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Gender</strong>: {st.session_state['patient_info']['gender']}</p>", unsafe_allow_html=True)
            st.markdown("---")
            
        # Navigation buttons
        if st.button("🏠 Welcome"):
            st.session_state['current_page'] = 'welcome'
            st.rerun()
            
        if st.button("👤 Patient Information"):
            st.session_state['current_page'] = 'patient_info'
            st.rerun()
            
        if st.button("📹 Emotion Recognition"):
            st.session_state['current_page'] = 'emotion_recognition'
            st.rerun()
            
        if st.button("📝 Questionnaire"):
            st.session_state['current_page'] = 'questionnaire'
            st.rerun()
            
        if st.button("📊 Assessment Results"):
            st.session_state['current_page'] = 'results'
            st.rerun()
            
        if st.button("📋 Clinical Report"):
            st.session_state['current_page'] = 'report'
            st.rerun()
            
        if st.button("👥 Patient Records"):
            st.session_state['current_page'] = 'patient_records'
            st.rerun()
        
        # System information
        st.markdown("---")
        st.markdown("<h3>System Information</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Model: EfficientNet-B0 + Transformer</p>", unsafe_allow_html=True)
        
        # Show device information with appropriate icon
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            st.markdown(f"<p>💻 Processing Device: <span style='color:#4CAF50; font-weight:bold;'>GPU - {device_name}</span></p>", unsafe_allow_html=True)
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            st.markdown(f"<p>💻 Processing Device: <span style='color:#2196F3; font-weight:bold;'>Apple MPS</span></p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p>💻 Processing Device: <span style='color:#9E9E9E;'>CPU</span> (GPU acceleration unavailable)</p>", unsafe_allow_html=True)

# --- Pages ---
def welcome_page():
    """Render the welcome page"""
    # Header with larger styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; color: #0D47A1; margin-bottom: 0;">Welcome to MindSight 🧠</h1>
        <p style="font-size: 1.3rem; color: #546E7A; font-style: italic; margin-top: 0;">Advanced Mental Health Assessment System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="card">
        <h3>About MindSight</h3>
        <p>MindSight is an advanced mental health assessment system that combines facial emotion recognition technology with standardized questionnaires to provide comprehensive mental health risk assessments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Process Flow Section
    st.markdown("<h2>Assessment Process</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center; background-color: #E3F2FD;">
            <h3 style="color: #1976D2;">1</h3>
            <p><b>Patient Information</b></p>
            <p>Record basic details about the patient</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center; background-color: #E8F5E9;">
            <h3 style="color: #4CAF50;">2</h3>
            <p><b>Facial Analysis</b></p>
            <p>Analyze facial expressions for emotional indicators</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="card" style="text-align: center; background-color: #FFF8E1;">
            <h3 style="color: #FFA000;">3</h3>
            <p><b>Questionnaire</b></p>
            <p>Complete standardized assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="card" style="text-align: center; background-color: #F3E5F5;">
            <h3 style="color: #9C27B0;">4</h3>
            <p><b>Results</b></p>
            <p>View integrated assessment results</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown("""
        <div class="card" style="text-align: center; background-color: #E0F7FA;">
            <h3 style="color: #00BCD4;">5</h3>
            <p><b>Clinical Report</b></p>
            <p>Generate detailed report with recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Privacy Notice
    st.markdown("""
    <div class="card" style="margin-top: 1.5rem;">
        <h3>Privacy Notice</h3>
        <p>All data is processed locally and not transmitted to external servers. Patient information and assessment results should be handled according to your institution's privacy policies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action Section
    st.markdown("""
    <div class="card" style="background-color: #FFECB3; margin-top: 1.5rem; text-align: center;">
        <h3>Ready to begin?</h3>
        <p>Start a new patient assessment or view existing patient records.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("New Patient Assessment", use_container_width=True):
            st.session_state['current_page'] = 'patient_info'
            st.rerun()
            
    with col2:
        if st.button("View Patient Records", use_container_width=True):
            st.session_state['current_page'] = 'patient_records'
            st.rerun()

def patient_info_page():
    """Render the patient information page"""
    st.markdown("<h1>Patient Information</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>Enter basic information about the patient before beginning the assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", value=st.session_state['patient_info']['id'])
            patient_name = st.text_input("Patient Name", value=st.session_state['patient_info']['name'])
        
        with col2:
            patient_age = st.text_input("Age", value=st.session_state['patient_info']['age'])
            patient_gender = st.selectbox("Gender", 
                options=["", "Male", "Female", "Non-binary", "Prefer not to say"],
                index=0 if not st.session_state['patient_info']['gender'] else 
                      ["", "Male", "Female", "Non-binary", "Prefer not to say"].index(st.session_state['patient_info']['gender']))
        
        patient_notes = st.text_area("Clinical Notes (Optional)", value=st.session_state['patient_info']['notes'], 
                                   height=150, max_chars=1000)
        
        submitted = st.form_submit_button("Save Patient Information")
        
    if submitted:
        st.session_state['patient_info'] = {
            'id': patient_id,
            'name': patient_name,
            'age': patient_age,
            'gender': patient_gender,
            'notes': patient_notes
        }
        
        st.success("Patient information saved successfully.")
        st.markdown("""
        <div class="card">
            <h3>Next Steps</h3>
            <p>Now proceed to the Facial Emotion Recognition assessment or Questionnaire.</p>
        </div>
        """, unsafe_allow_html=True)

def emotion_recognition_page():
    """Render the emotion recognition page"""
    st.markdown("<h1>Facial Emotion Recognition</h1>", unsafe_allow_html=True)
    
    if not st.session_state['patient_info']['name']:
        st.warning("Please enter patient information first before proceeding with the assessment.")
        st.button("Go to Patient Information", on_click=lambda: setattr(st.session_state, 'current_page', 'patient_info'))
        return
    
    st.markdown("""
    <div class="card">
        <h3>Instructions</h3>
        <p>This assessment will use your webcam to analyze facial expressions in real-time. The system will detect emotions including happiness, sadness, anger, fear, disgust, surprise, and neutral expressions.</p>
        <p>Please ensure the patient is:</p>
        <ul>
            <li>Well-lit from the front</li>
            <li>Looking directly at the camera</li>
            <li>Positioned in a neutral environment</li>
        </ul>
        <p>The session will run for approximately 60-90 seconds to gather sufficient data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add troubleshooting information in expandable section
    with st.expander("Camera Troubleshooting (Click to expand)"):
        st.markdown("""
        ### Camera Not Working?
        
        1. **Check Camera Connection**:
           - Make sure your webcam is properly connected to your computer
           - Try testing your camera in another application like Photo Booth (Mac) or Camera (Windows)
           - Ensure no other applications are currently using the camera
        
        2. **Docker Issues**:
           - When running in Docker, make sure to properly map the camera device
           - Add `--device=/dev/video0:/dev/video0` to your Docker run command
           - In Docker Compose, add: `devices: ["/dev/video0:/dev/video0"]`
           - On Mac, you may need to install a virtual camera device
        
        3. **Still Not Working?**:
           - Try running the app directly without Docker: `streamlit run src/dashboard_app.py`
           - Restart your computer to reset camera connections
           - Check if you need to update your webcam drivers
        """)
    
    model_path = os.environ.get("MODEL_PATH", "checkpoints/model_dataaugmented/best_model.pth")
    
    # Initialize webcam control variables
    if 'running' not in st.session_state:
        st.session_state['running'] = False
    
    # Start/Stop buttons
    col1, col2 = st.columns([1, 3])
    with col1:
        if not st.session_state['running']:
            start_button = st.button("▶️ Start Session")
            if start_button:
                st.session_state['running'] = True
                st.session_state['emotion_history'] = []  # Reset emotion history
                st.session_state['webcam_active'] = True
                st.rerun()
        else:
            stop_button = st.button("⏹️ Stop Session")
            if stop_button:
                st.session_state['running'] = False
                st.session_state['webcam_active'] = False
                st.rerun()
    
    # Webcam placeholder
    stframe = st.empty()
    
    # Progress placeholder
    progress_placeholder = st.empty()
    
    # Results area
    results_area = st.empty()
    
    # Using OpenCV for webcam capture
    if st.session_state['running']:
        try:
            model, device = load_fer_model(model_path)
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("""
                Error: Could not access webcam. 
                
                Troubleshooting tips:
                1. Make sure your camera is not being used by another application
                2. Check that your camera is properly connected and working
                3. When running in Docker, make sure to properly map the camera device
                4. Try running the application outside Docker to test if the camera works
                5. See webcam-setup.md for detailed instructions
                """)
                st.session_state['running'] = False
            else:
                emotion_history = []
                frame_count = 0
                max_frames = 120  # Approximately 2 minutes at 1 FPS
                
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Main processing loop
                while st.session_state['running'] and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam disconnected.")
                        break
                    
                    # Update progress
                    progress_percent = int((frame_count / max_frames) * 100)
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Recording... {progress_percent}% complete")
                    
                    # Process every 2nd frame to reduce CPU load
                    if frame_count % 2 == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Create a face cascade classifier
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        
                        # Try with different parameters to increase detection chances
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                        
                        # If no faces detected, use center region as a fallback
                        if len(faces) == 0:
                            h, w = frame.shape[:2]
                            center_x = w // 4
                            center_y = h // 4
                            center_w = w // 2
                            center_h = h // 2
                            faces = np.array([[center_x, center_y, center_w, center_h]])
                            # Add a note that we're using fallback detection
                            cv2.putText(frame, "Using fallback face region", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        for (x, y, w, h) in faces:
                            face = frame[y:y+h, x:x+w]
                            input_tensor = preprocess_face(face).to(device)
                            
                            with torch.no_grad():
                                output = model(input_tensor)
                                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                                emotion_idx = np.argmax(probs)
                                emotion_label = EMOTIONS[emotion_idx]
                                emotion_history.append(emotion_label)
                            
                            # Draw rectangle around face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 120, 255), 2)
                            
                            # Draw emotion label with confidence
                            label = f"{emotion_label}: {probs[emotion_idx]:.2f}"
                            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            y_label = max(y - 10, label_size[1])
                            cv2.putText(frame, label, (x, y_label), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
                    
                    # Display the frame
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    frame_count += 1
                    
                    # Slow down capture rate
                    cv2.waitKey(25)  # Add small delay
                
                # Save results when done
                cap.release()
                
                if emotion_history:
                    st.session_state['emotion_history'] = emotion_history
                    st.session_state['running'] = False
                    progress_placeholder.empty()
                    st.success("Session completed successfully!")
                    st.rerun()
        
        except Exception as e:
            st.error(f"An error occurred during the emotion recognition session: {str(e)}")
            st.session_state['running'] = False
    
    # Show results if available
    if not st.session_state['running'] and st.session_state['emotion_history']:
        with results_area.container():
            st.markdown("""
            <div class="card">
                <h3>Session Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Emotion Distribution")
                emotion_chart = generate_emotion_chart(st.session_state['emotion_history'])
                if emotion_chart:
                    st.pyplot(emotion_chart)
            
            with col2:
                st.subheader("Emotion Trends")
                emotion_time_chart = generate_emotion_time_series(st.session_state['emotion_history'])
                if emotion_time_chart:
                    st.pyplot(emotion_time_chart)
            
            # Generate insights
            emotion_insights, emotion_risk_score = analyze_emotion_patterns(st.session_state['emotion_history'])
            
            st.markdown("<h3>Detected Patterns & Insights</h3>", unsafe_allow_html=True)
            for insight in emotion_insights:
                st.markdown(f"- {insight}")
            
            # Next steps
            st.markdown("""
            <div class="card">
                <h3>Next Steps</h3>
                <p>Please proceed to the Questionnaire assessment to complete the evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Continue to Questionnaire"):
                st.session_state['current_page'] = 'questionnaire'
                st.rerun()

def questionnaire_page():
    """Render the questionnaire assessment page"""
    st.markdown("<h1>Mental Health Assessment Questionnaire</h1>", unsafe_allow_html=True)
    
    if not st.session_state['patient_info']['name']:
        st.warning("Please enter patient information first before proceeding with the assessment.")
        st.button("Go to Patient Information", on_click=lambda: setattr(st.session_state, 'current_page', 'patient_info'))
        return
    
    st.markdown("""
    <div class="card">
        <h3>Instructions</h3>
        <p>This questionnaire is based on standardized mental health screening tools including PHQ-9 (depression) and GAD-7 (anxiety).</p>
        <p>Please answer each question based on the patient's experiences over the past 2 weeks.</p>
        <p>The combined results will help assess overall mental health risk level.</p>
    </div>
    """, unsafe_allow_html=True)
    
    questions, scale = load_questionnaire()
    responses = []
    
    with st.form("clinical_assessment_form"):
        st.markdown("<h3>Over the last 2 weeks, how often has the patient been bothered by the following problems?</h3>", unsafe_allow_html=True)
        
        for i, q in enumerate(questions):
            st.markdown(f"<p><strong>Q{i+1}: {q}</strong></p>", unsafe_allow_html=True)
            answer = st.radio("Select the most appropriate response:", 
                              list(scale.keys()), 
                              format_func=lambda x: f"{scale[x]}", 
                              key=f"q{i}",
                              horizontal=True)
            responses.append(int(answer))
        
        submitted = st.form_submit_button("Submit Assessment")
    
    if submitted:
        score = sum(responses)
        
        if score < 5:
            result = "Minimal Risk"
            result_class = "info-minimal"
        elif score < 10:
            result = "Mild Risk"
            result_class = "info-mild"
        elif score < 15:
            result = "Moderate Risk"
            result_class = "info-moderate"
        else:
            result = "High Risk"
            result_class = "info-high"
        
        summary = {
            "responses": responses,
            "total_score": score,
            "risk_level": result
        }
        
        st.session_state['questionnaire_results'] = summary
        
        # Show results
        st.markdown(f"""
        <div class="card {result_class}">
            <h3>Assessment Results</h3>
            <p><strong>Total Score:</strong> {score}/21</p>
            <p><strong>Risk Level:</strong> {result}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate risk visualization
        risk_gauge = generate_risk_gauge(score)
        st.pyplot(risk_gauge)
        
        # Check if both assessments are complete
        if st.session_state['emotion_history']:
            st.session_state['assessment_complete'] = True
            st.markdown("""
            <div class="card">
                <h3>Next Steps</h3>
                <p>Both assessments are now complete. Please proceed to the Assessment Results page.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card">
                <h3>Next Steps</h3>
                <p>Please complete the Facial Emotion Recognition assessment to generate a comprehensive report.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Go to Emotion Recognition"):
                st.session_state['current_page'] = 'emotion_recognition'
                st.rerun()

def results_page():
    """Render the results and recommendations page"""
    st.markdown("<h1>Assessment Results & Recommendations</h1>", unsafe_allow_html=True)
    
    if not st.session_state['patient_info']['name']:
        st.warning("Please enter patient information first before proceeding with the assessment.")
        st.button("Go to Patient Information", on_click=lambda: setattr(st.session_state, 'current_page', 'patient_info'))
        return
    
    # Check if both assessments are complete
    emotion_complete = bool(st.session_state['emotion_history'])
    questionnaire_complete = bool(st.session_state['questionnaire_results'])
    
    st.markdown("""
    <div class="card">
        <h3>Assessment Status</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if emotion_complete:
            st.success("✓ Emotion Recognition Assessment: Complete")
        else:
            st.warning("⚠ Emotion Recognition Assessment: Incomplete")
    
    with col2:
        if questionnaire_complete:
            st.success("✓ Questionnaire Assessment: Complete")
        else:
            st.warning("⚠ Questionnaire Assessment: Incomplete")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # If both assessments are complete, show combined results
    if emotion_complete and questionnaire_complete:
        # Generate insights from emotion analysis
        emotion_insights, emotion_risk_score = analyze_emotion_patterns(st.session_state['emotion_history'])
        questionnaire_score = st.session_state['questionnaire_results']['total_score']
        
        # Calculate combined risk
        risk_level, risk_score = calculate_combined_risk(questionnaire_score, emotion_risk_score)
        st.session_state['combined_assessment'] = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'emotion_insights': emotion_insights
        }
        
        # Determine the risk class for CSS styling
        risk_class = ""
        if "Minimal" in risk_level:
            risk_class = "info-minimal"
        elif "Mild" in risk_level:
            risk_class = "info-mild"
        elif "Moderate" in risk_level:
            risk_class = "info-moderate"
        else:
            risk_class = "info-high"
        
        # Display combined risk assessment
        st.markdown(f"""
        <div class="card {risk_class}">
            <h2>Combined Risk Assessment</h2>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p><strong>Integrated Risk Score:</strong> {risk_score:.1f}/10</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Emotion Analysis</h3>", unsafe_allow_html=True)
            emotion_chart = generate_emotion_chart(st.session_state['emotion_history'])
            if emotion_chart:
                st.pyplot(emotion_chart)
                
            st.markdown("#### Key Insights")
            for insight in emotion_insights:
                st.markdown(f"- {insight}")
        
        with col2:
            st.markdown("<h3>Questionnaire Results</h3>", unsafe_allow_html=True)
            risk_gauge = generate_risk_gauge(questionnaire_score)
            st.pyplot(risk_gauge)
            
            st.markdown("#### Questionnaire Highlights")
            responses = st.session_state['questionnaire_results']['responses']
            questions, scale = load_questionnaire()
            
            # Highlight significant responses (score >= 2)
            significant_responses = [(i, q, r) for i, (q, r) in enumerate(zip(questions, responses)) if r >= 2]
            
            if significant_responses:
                for i, question, response in significant_responses:
                    st.markdown(f"- **Q{i+1}:** {question} - **Response:** {scale[str(response)]}")
            else:
                st.markdown("- No significant concerns identified in questionnaire")
        
        # Coping strategies and recommendations
        st.markdown("---")
        st.markdown("<h2>Clinical Recommendations</h2>", unsafe_allow_html=True)
        
        coping_strategies = get_coping_strategies(risk_level, emotion_insights)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3>Suggested Interventions</h3>", unsafe_allow_html=True)
            for i, strategy in enumerate(coping_strategies):
                st.markdown(f"- {strategy}")
        
        with col2:
            st.markdown("<h3>Follow-up Plan</h3>", unsafe_allow_html=True)
            
            if "Minimal" in risk_level:
                st.markdown("""
                - Routine follow-up in 3 months
                - Self-monitoring recommended
                - Provide educational resources
                """)
            elif "Mild" in risk_level:
                st.markdown("""
                - Follow-up in 4-6 weeks
                - Consider low-intensity interventions
                - Provide self-help resources
                """)
            elif "Moderate" in risk_level:
                st.markdown("""
                - Follow-up in 2-3 weeks
                - Consider referral to mental health services
                - Regular check-ins recommended
                """)
            else:  # High risk
                st.markdown("""
                - Prompt follow-up within 1 week
                - Immediate referral to specialized services
                - Consider safety planning if indicated
                """)
        
        st.markdown("""
        <div class="card">
            <h3>Next Steps</h3>
            <p>You can now generate a complete clinical report based on the assessment results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Clinical Report"):
            st.session_state['current_page'] = 'report'
            st.rerun()
    
    else:
        # If assessments are incomplete, prompt to complete them
        st.markdown("""
        <div class="card">
            <h3>Complete Assessments</h3>
            <p>Please complete both assessments to view combined results and recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not emotion_complete:
                if st.button("Complete Emotion Recognition"):
                    st.session_state['current_page'] = 'emotion_recognition'
                    st.rerun()
        
        with col2:
            if not questionnaire_complete:
                if st.button("Complete Questionnaire"):
                    st.session_state['current_page'] = 'questionnaire'
                    st.rerun()

def report_page():
    """Render the clinical report page"""
    st.markdown("<h1>Clinical Assessment Report</h1>", unsafe_allow_html=True)
    
    if not st.session_state['patient_info']['name']:
        st.warning("Please enter patient information first before proceeding.")
        st.button("Go to Patient Information", on_click=lambda: setattr(st.session_state, 'current_page', 'patient_info'))
        return
    
    # Check if all assessments are complete
    if not st.session_state['emotion_history'] or not st.session_state['questionnaire_results']:
        st.warning("Please complete all assessments before generating a report.")
        st.button("Go to Results", on_click=lambda: setattr(st.session_state, 'current_page', 'results'))
        return
    
    # Check if combined assessment is missing
    if st.session_state['combined_assessment'] is None:
        st.warning("Please view the Assessment Results page first to generate combined assessment.")
        st.button("Go to Results", on_click=lambda: setattr(st.session_state, 'current_page', 'results'))
        return
    
    # Clinical report
    patient = st.session_state['patient_info']
    combined = st.session_state['combined_assessment']
    emotion_insights, _ = analyze_emotion_patterns(st.session_state['emotion_history'])
    questionnaire = st.session_state['questionnaire_results']
    
    report_date = datetime.now().strftime("%B %d, %Y")
    
    # Generate report content
    st.markdown("""
    <div class="card">
        <div style="text-align:center">
            <h2>CLINICAL ASSESSMENT REPORT</h2>
            <p>Generated by MindSight Assessment System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="report-section">
            <div class="report-header">Patient Information</div>
            <div class="report-item"><strong>Patient ID:</strong> {patient['id']}</div>
            <div class="report-item"><strong>Name:</strong> {patient['name']}</div>
            <div class="report-item"><strong>Age:</strong> {patient['age']}</div>
            <div class="report-item"><strong>Gender:</strong> {patient['gender']}</div>
            <div class="report-item"><strong>Assessment Date:</strong> {report_date}</div>
        </div>
        
        <div class="report-section">
            <div class="report-header">Assessment Results</div>
            <div class="report-item"><strong>Combined Risk Level:</strong> {combined['risk_level']}</div>
            <div class="report-item"><strong>Risk Score:</strong> {combined['risk_score']:.1f}/10</div>
            <div class="report-item"><strong>Questionnaire Score:</strong> {questionnaire['total_score']}/21</div>
        </div>
        
        <div class="report-section">
            <div class="report-header">Facial Emotion Analysis</div>
    """, unsafe_allow_html=True)
    
    # Generate emotion distribution counts
    emotion_counts = Counter(st.session_state['emotion_history'])
    total_frames = len(st.session_state['emotion_history'])
    
    for emotion, count in emotion_counts.most_common():
        percentage = (count / total_frames) * 100
        st.markdown(f"""
            <div class="report-item"><strong>{emotion.title()}:</strong> {count} frames ({percentage:.1f}%)</div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
        
        <div class="report-section">
            <div class="report-header">Emotional Insights</div>
    """, unsafe_allow_html=True)
    
    for insight in emotion_insights:
        st.markdown(f"""
            <div class="report-item">• {insight}</div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
        
        <div class="report-section">
            <div class="report-header">Clinical Recommendations</div>
    """, unsafe_allow_html=True)
    
    coping_strategies = get_coping_strategies(combined['risk_level'], emotion_insights)
    for strategy in coping_strategies:
        st.markdown(f"""
            <div class="report-item">• {strategy}</div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
        
        <div class="report-section">
            <div class="report-header">Follow-up Plan</div>
    """, unsafe_allow_html=True)
    
    if "Minimal" in combined['risk_level']:
        st.markdown("""
            <div class="report-item">• Routine follow-up in 3 months</div>
            <div class="report-item">• Self-monitoring recommended</div>
            <div class="report-item">• Provide educational resources</div>
        """, unsafe_allow_html=True)
    elif "Mild" in combined['risk_level']:
        st.markdown("""
            <div class="report-item">• Follow-up in 4-6 weeks</div>
            <div class="report-item">• Consider low-intensity interventions</div>
            <div class="report-item">• Provide self-help resources</div>
        """, unsafe_allow_html=True)
    elif "Moderate" in combined['risk_level']:
        st.markdown("""
            <div class="report-item">• Follow-up in 2-3 weeks</div>
            <div class="report-item">• Consider referral to mental health services</div>
            <div class="report-item">• Regular check-ins recommended</div>
        """, unsafe_allow_html=True)
    else:  # High risk
        st.markdown("""
            <div class="report-item">• Prompt follow-up within 1 week</div>
            <div class="report-item">• Immediate referral to specialized services</div>
            <div class="report-item">• Consider safety planning if indicated</div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
        
        <div class="report-section">
            <div class="report-header">Clinician Notes</div>
            <div class="clinical-notes">
    """, unsafe_allow_html=True)
    
    if patient['notes']:
        st.markdown(f"{patient['notes']}", unsafe_allow_html=True)
    else:
        st.markdown("No additional notes provided.", unsafe_allow_html=True)
    
    # Close the previous markdown section
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add signature section separately
    st.markdown("""
        <div class="report-section" style="margin-top:30px;text-align:center">
            <hr style="width:70%; margin:0 auto;">
            <p style="margin-top:10px;">Clinician Signature</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add footer separately
    st.markdown("""
        <div style="margin-top:20px;text-align:center;font-size:0.8rem;color:#616161">
            <p>This report was generated using the MindSight Clinical Assessment System.</p>
            <p>Results should be interpreted by qualified healthcare professionals.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization export
    st.markdown("---")
    st.markdown("<h2>Report Visualizations</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Emotion Distribution")
        emotion_chart = generate_emotion_chart(st.session_state['emotion_history'])
        if emotion_chart:
            st.pyplot(emotion_chart)
    
    with col2:
        st.subheader("Risk Assessment")
        risk_gauge = generate_risk_gauge(questionnaire['total_score'])
        st.pyplot(risk_gauge)
    
    # Export options
    st.markdown("---")
    st.markdown("<h2>Export Options</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("Export as PDF", disabled=True)
    
    with col2:
        st.button("Print Report", disabled=True)
    
    with col3:
        st.button("Save to Patient Records", disabled=True)
    
    st.info("Note: Export functionality is not implemented in this demo version.")
    
    # Patient Management
    st.markdown("---")
    st.markdown("<h2>Patient Management</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Start New Assessment</h3>
            <p>Begin a new assessment with a different patient</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("New Patient Assessment"):
            # Reset session state for new patient
            st.session_state['patient_info'] = {
                'id': '',
                'name': '',
                'age': '',
                'gender': '',
                'notes': ''
            }
            st.session_state['emotion_history'] = []
            st.session_state['questionnaire_results'] = None
            st.session_state['combined_assessment'] = None
            st.session_state['assessment_complete'] = False
            st.session_state['current_page'] = 'patient_info'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Patient Records</h3>
            <p>View or manage previous patient assessments</p>
        </div>
        """, unsafe_allow_html=True)
        
        # In a real implementation, this would load patients from a database
        # For demo purposes, we'll just show an example interface
        if st.button("Manage Patient Records"):
            st.session_state['current_page'] = 'patient_records'
            st.rerun()

# --- Patient Records Management ---
def patient_records_page():
    """Render the patient records management page"""
    st.markdown("<h1>Patient Records Management</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Patient Records Database</h3>
        <p>In a production environment, this page would allow clinicians to:</p>
        <ul>
            <li>Search for patient records</li>
            <li>View patient assessment history</li>
            <li>Compare assessments over time</li>
            <li>Export patient data</li>
            <li>Manage data retention policies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Example patient records (in production, this would come from a database)
    example_patients = [
        {"id": "PT-2025-001", "name": "John Smith", "age": "45", "last_visit": "May 12, 2025", "risk_level": "Minimal Risk"},
        {"id": "PT-2025-014", "name": "Emily Johnson", "age": "32", "last_visit": "May 18, 2025", "risk_level": "Mild Risk"},
        {"id": "PT-2025-022", "name": "Michael Brown", "age": "51", "last_visit": "May 20, 2025", "risk_level": "Moderate Risk"},
        {"id": "PT-2025-031", "name": "Sarah Wilson", "age": "29", "last_visit": "May 23, 2025", "risk_level": "High Risk"},
    ]
    
    st.markdown("<h2>Recent Patients</h2>", unsafe_allow_html=True)
    
    # Create a DataFrame from the example patients for better display
    import pandas as pd
    
    # Convert patient data to DataFrame
    df = pd.DataFrame(example_patients)
    
    # Add styling to the risk level column
    def color_risk_level(val):
        if "Minimal" in val:
            return f'background-color: #E8F5E9; color: #4CAF50; font-weight: bold'
        elif "Mild" in val:
            return f'background-color: #FFF3E0; color: #FFC107; font-weight: bold'
        elif "Moderate" in val:
            return f'background-color: #FBE9E7; color: #FF9800; font-weight: bold'
        elif "High" in val:
            return f'background-color: #FFEBEE; color: #F44336; font-weight: bold'
        return ''
    
    # Rename columns for display
    df.columns = ["Patient ID", "Name", "Age", "Last Assessment", "Risk Level"]
    
    # Display the styled table
    styled_df = df.style.applymap(color_risk_level, subset=["Risk Level"])
    st.dataframe(styled_df, use_container_width=True)
    
    # Search functionality demo
    st.markdown("<h2>Search Records</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.text_input("Patient ID", placeholder="Enter patient ID...")
    
    with col2:
        st.text_input("Patient Name", placeholder="Enter patient name...")
    
    with col3:
        st.selectbox("Risk Level", ["All Levels", "Minimal Risk", "Mild Risk", "Moderate Risk", "High Risk"])
    
    if st.button("Search Records"):
        st.info("Search functionality is not implemented in this demo version.")
    
    st.markdown("---")
    
    # Return to dashboard
    if st.button("Return to Welcome Page"):
        st.session_state['current_page'] = 'welcome'
        st.rerun()

# --- Main App Logic ---
def main():
    """Main function to run the app"""
    render_header()
    render_sidebar()
    
    # Render the appropriate page based on current state
    if st.session_state['current_page'] == 'welcome':
        welcome_page()
    elif st.session_state['current_page'] == 'patient_info':
        patient_info_page()
    elif st.session_state['current_page'] == 'emotion_recognition':
        emotion_recognition_page()
    elif st.session_state['current_page'] == 'questionnaire':
        questionnaire_page()
    elif st.session_state['current_page'] == 'results':
        results_page()
    elif st.session_state['current_page'] == 'report':
        report_page()
    elif st.session_state['current_page'] == 'patient_records':
        patient_records_page()

if __name__ == "__main__":
    main()

