import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
ATTENDANCE_LOGS_DIR = STORAGE_DIR / "attendance_logs"
ANALYTICS_DIR = STORAGE_DIR / "session_analytics"
ROSTER_DIR = STORAGE_DIR / "roster"
UPLOADS_DIR = STORAGE_DIR / "uploads"

# Ensure runtime directories exist
for path in [STORAGE_DIR, ATTENDANCE_LOGS_DIR, ANALYTICS_DIR, ROSTER_DIR, UPLOADS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Application Settings
SECRET_KEY = os.getenv("SECRET_KEY", "multimodal-classroom-secret-key-2026")
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Video & Camera Processing
DEFAULT_CAMERA_INDEX = 0
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
TARGET_FPS = 15
PROCESS_EVERY_NTH_FRAME = 2  # Balance real-time performance and CPU efficiency

# Face Detection & Tracking Configuration
MIN_FACE_CONFIDENCE = 0.5
IOU_TRACKER_THRESHOLD = 0.35  # Threshold for matching tracked bounding boxes across frames
MAX_TRACK_AGE_FRAMES = 30     # How long to keep a lost track alive before removing

# Sustained Presence Logic (Attendance Validation)
# To eliminate false positives from passersby, presence must be sustained
SUSTAINED_PRESENCE_WINDOW_SEC = 30.0  # Sliding observation window (seconds)
MIN_DETECTION_STREAK_SEC = 3.0        # Consecutive seconds to consider a candidate active
MIN_ATTENDANCE_RATIO_PRESENT = 0.60   # >= 60% session sustained presence -> Present
MIN_ATTENDANCE_RATIO_PARTIAL = 0.30   # 30% - 59% session sustained presence -> Partial
# < 30% sustained presence -> Absent

# Engagement Analysis Parameters
# Head pose thresholds (in degrees)
HEAD_POSE_PITCH_DOWN = -18.0  # Slouching / looking down at desk or phone
HEAD_POSE_PITCH_UP = 22.0     # Looking up at ceiling
HEAD_POSE_YAW_THRESHOLD = 25.0 # Looking sideways away from board/instructor
HEAD_POSE_ROLL_THRESHOLD = 20.0 # Extreme head tilt

# Gaze and Inactivity
EYE_ASPECT_RATIO_DROWSY = 0.20 # Below this indicates eye closure / drowsy
MAX_INACTIVITY_SEC = 25.0      # Slump / totally motionless threshold

# Audio Processing Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
AUDIO_RMS_SILENCE_THRESHOLD = 0.015
AUDIO_RMS_DISCUSSION_THRESHOLD = 0.075
AUDIO_SPEECH_FREQ_MIN = 85     # Hz (human voice range lower)
AUDIO_SPEECH_FREQ_MAX = 3000   # Hz (human voice range upper)

# Multimodal Fusion Weights (Total = 1.0)
FUSION_WEIGHT_VISUAL_ATTENTION = 0.55
FUSION_WEIGHT_MOVEMENT_ALERTNESS = 0.15
FUSION_WEIGHT_AUDIO_INTERACTION = 0.30

# AI Teaching Suggestion Engine Thresholds
ALERT_ENGAGEMENT_DROP_THRESHOLD = 45.0  # Score below 45/100 triggers attention alert
ALERT_PROLONGED_SILENCE_SEC = 180.0     # 3 minutes of total silence -> prompt interactive question
ALERT_HIGH_RESTLESSNESS_RATIO = 0.40    # 40% of class restless -> suggestion for stretch/break
ALERT_HIGH_DROWSINESS_RATIO = 0.30      # 30% of class drowsy -> quick engagement activity

# Privacy & Security
ENABLE_OFFLINE_MODE = True
ANONYMIZE_FACES_DEFAULT = False  # Can blur faces for high-privacy compliance
SAVE_RAW_FRAMES_TO_DISK = False  # Privacy safeguard: Never persist raw student frames without consent
