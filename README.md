# 🚗 Traffic Detection & Tracking System

A production-grade real-time traffic analysis system using YOLOv8, ByteTrack, and OpenCV. Detects, tracks, and analyzes vehicles with speed estimation, direction detection, and comprehensive counting logic.

## ✨ Features

- **Real-time Detection**: YOLOv8s for vehicle detection (cars, motorcycles, buses, trucks)
- **SOTA Tracking**: ByteTrack algorithm for robust multi-object tracking
- **Speed Estimation**: Pixel-to-meter calibration with perspective transform for accurate speed calculation
- **Vehicle Counting**: Intelligent counting with direction detection (approaching/leaving)
- **Direction Analysis**: Determines vehicle movement direction (approaching or departing)
- **Trace Visualization**: Historical trajectory visualization for each tracked vehicle
- **HUD Metrics**: Real-time display of vehicle count, active tracks, and FPS
- **Analytics Export**: Frame-by-frame metrics collection for post-analysis
- **Camera Calibration**: Perspective-aware distance calculation

## 📋 Requirements

- Python 3.8+
- OpenCV 4.5+
- YOLOv8 (ultralytics)
- supervision
- NumPy

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-detection-tracking.git
cd traffic-detection-tracking

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from traffic_analyzer import TrafficAnalyzer, CameraCalibration
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

# Initialize calibration
calibration = CameraCalibration(pixel_to_meter_ratio=0.035)
calibration.auto_init_from_frame(1920, 1080)

# Create analyzer
analyzer = TrafficAnalyzer(model, fps=30, calibration=calibration)

# Process video
cap = cv2.VideoCapture("input_video.mp4")
writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated, analytics = analyzer.process_frame(frame, frame_num)
    writer.write(annotated)

cap.release()
writer.release()
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5

# Tracking
TRACK_ACTIVATION_THRESHOLD = 0.25
LOST_TRACK_BUFFER = 30
MINIMUM_MATCHING_THRESHOLD = 0.8
MINIMUM_CONSECUTIVE_FRAMES = 3

# Calibration
PIXEL_TO_METER_RATIO = 0.035

# Counting Line
COUNT_LINE_Y_RATIO = 0.6  # 60% from top
```

## 📊 Classes

### `CameraCalibration`
Handles pixel-to-meter conversion with perspective transform for accurate distance and speed calculations.

**Methods:**
- `auto_init_from_frame(frame_w, frame_h)` - Initialize from frame dimensions
- `pixel_to_meter(pixel_dist, y_position)` - Convert pixels to meters
- `transform_point(x, y)` - Transform point to bird's eye view

### `TrafficAnalyzer`
Main analysis engine combining detection, tracking, and analytics.

**Methods:**
- `process_frame(frame, frame_num)` - Full pipeline (detect → track → analyze → annotate)
- `estimate_speed(track_id, current_pos, frame_num)` - Calculate vehicle speed (km/h)
- `detect_direction(track_id, current_y)` - Determine approach/leaving direction
- `count_vehicle(track_id, centre_y, direction)` - Count vehicles crossing counting line

## 📈 Output

The system generates:
1. **Annotated Video**: Tracked vehicles with bounding boxes, speeds, and IDs
2. **Analytics JSON**: Frame-by-frame metrics
3. **Console Metrics**: Real-time FPS, count, and processing speed

## 🔧 Speed Estimation Details

- Uses 10-frame history for stability
- Applies Kalman smoothing via median filtering
- Accounts for perspective distortion (closer vehicles = larger pixel movement)
- Converts to km/h from pixel distance
- Sanity check: rejects speeds > 200 km/h

## 🚗 Vehicle Classes

- `2` - Car
- `3` - Motorcycle  
- `5` - Bus
- `7` - Truck

## 📝 Example Output Metrics

```json
{
  "frame": 150,
  "timestamp": 5.0,
  "vehicle_count": 12,
  "active_tracks": 5,
  "speeds": {
    "1": 45.3,
    "2": 62.1,
    "3": null
  },
  "directions": {
    "1": "approaching",
    "2": "leaving",
    "3": "unknown"
  }
}
```

## 🎯 Use Cases

- **Traffic Monitoring**: Real-time vehicle counting and speed analysis
- **Road Safety**: Detecting speeding vehicles
- **Traffic Flow Analysis**: Understanding congestion patterns
- **Retail Analytics**: Parking lot vehicle tracking
- **Security Surveillance**: Vehicle detection and tracking

## 🐛 Known Limitations

- Speed estimation accuracy depends on camera calibration
- Performance degrades with heavy occlusion
- Requires GPU for real-time processing (30+ FPS)
- ByteTrack may lose tracks during long occlusions

## 💡 Future Improvements

- [ ] Lane detection and traffic direction per lane
- [ ] Anomaly detection (wrong-way vehicles, stopped vehicles)
- [ ] Multi-camera tracking
- [ ] Adaptive calibration
- [ ] Real-time dashboard with Streamlit
- [ ] REST API for deployment

## 📜 License

MIT License - see LICENSE file

## 👤 Author

Krrish Jain

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Contact

For questions, issues, or suggestions, open a GitHub issue.

---

**⭐ If you found this useful, please star the repository!**
