"""
Configuration settings for Traffic Detection & Tracking System
"""

# ==================== MODEL SETTINGS ====================
MODEL_NAME = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5

# ==================== TRACKING SETTINGS ====================
TRACK_ACTIVATION_THRESHOLD = 0.25
LOST_TRACK_BUFFER = 30
MINIMUM_MATCHING_THRESHOLD = 0.8
FRAME_RATE = 30
MINIMUM_CONSECUTIVE_FRAMES = 3

# ==================== CAMERA CALIBRATION ====================
PIXEL_TO_METER_RATIO = 0.035  # meters per pixel
PERSPECTIVE_SAMPLE_MARGIN = {
    'top_left': 0.2,      # x: 20% from left
    'top_right': 0.8,     # x: 80% from left
    'bottom_left': 0.1,   # x: 10% from left
    'bottom_right': 0.9,  # x: 90% from left
    'top_y': 0.6,         # y: 60% from top
    'bottom_y': 0.95      # y: 95% from top
}

# ==================== COUNTING SETTINGS ====================
COUNT_LINE_Y_RATIO = 0.6  # Counting line at 60% from top

# ==================== SPEED ESTIMATION ====================
SPEED_HISTORY_WINDOW = 10  # frames to use for speed estimation
SPEED_SMOOTHING_WINDOW = 10  # frames for median filtering
SPEED_MAX_THRESHOLD = 200  # km/h - reject speeds above this
SPEED_MIN_TIME_DIFF = 0.3  # seconds - minimum time difference for estimation

# ==================== VEHICLE CLASSES ====================
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# ==================== VISUALIZATION ====================
BOX_THICKNESS = 2
LABEL_TEXT_SCALE = 0.5
LABEL_TEXT_THICKNESS = 1
TRACE_LENGTH = 60
TRACE_THICKNESS = 2
HUD_BG_ALPHA = 0.3

# ==================== OUTPUT SETTINGS ====================
OUTPUT_VIDEO_CODEC = 'mp4v'
SAVE_ANALYTICS = True
ANALYTICS_EXPORT_FORMAT = 'json'  # 'json' or 'csv'

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE = 'traffic_analysis.log'
VERBOSE = True
