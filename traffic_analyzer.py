"""
Traffic Detection and Tracking System
Production-grade implementation with YOLOv8 + ByteTrack + OpenCV
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Tuple, Optional, Dict, List
import logging
import json
from config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CameraCalibration:
    """
    Handles perspective transform and pixel-to-meter conversion.
    Converts pixel distances into real-world meters accounting for camera perspective.
    """

    def __init__(self, pixel_to_meter_ratio: Optional[float] = None):
        """
        Initialize camera calibration.
        
        Args:
            pixel_to_meter_ratio: Meters per pixel. Default: 0.04 (typical highway camera)
        """
        self.pixel_to_meter_ratio = pixel_to_meter_ratio or PIXEL_TO_METER_RATIO
        self.src_points = None
        self.dst_points = None
        self.perspective_matrix = None
        self.inv_matrix = None
        logger.info(f"CameraCalibration initialized with ratio: {self.pixel_to_meter_ratio}")

    def auto_init_from_frame(self, frame_w: int, frame_h: int) -> None:
        """
        Auto-initialize perspective transform for typical traffic camera setup.
        
        Args:
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
        """
        try:
            # Source: trapezoid region in image (road area)
            self.src_points = np.float32([
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['top_left'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['top_y']],
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['top_right'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['top_y']],
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['bottom_right'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['bottom_y']],
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['bottom_left'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['bottom_y']]
            ])

            # Destination: rectangle in bird's eye view
            self.dst_points = np.float32([
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['top_left'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['top_y']],
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['top_right'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['top_y']],
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['top_right'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['bottom_y']],
                [frame_w * PERSPECTIVE_SAMPLE_MARGIN['top_left'], frame_h * PERSPECTIVE_SAMPLE_MARGIN['bottom_y']]
            ])

            self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            self.inv_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
            logger.info(f"Camera calibration initialized for {frame_w}x{frame_h}")
        except Exception as e:
            logger.error(f"Error initializing camera calibration: {e}")
            raise

    def pixel_to_meter(self, pixel_dist: float, y_position: float) -> float:
        """
        Convert pixel distance to meters with perspective correction.
        
        Args:
            pixel_dist: Distance in pixels
            y_position: Y position in frame (0-1 normalized)
            
        Returns:
            Distance in meters
        """
        try:
            # Perspective correction: closer objects (larger y) have larger pixels per meter
            scale = 1.0 + (1.0 - y_position) * 0.5
            return pixel_dist * self.pixel_to_meter_ratio * scale
        except Exception as e:
            logger.error(f"Error in pixel_to_meter conversion: {e}")
            return pixel_dist * self.pixel_to_meter_ratio

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform point to bird's eye view.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Transformed (x, y) coordinates
        """
        try:
            if self.perspective_matrix is not None:
                pt = cv2.perspectiveTransform(
                    np.array([[[x, y]]], dtype=np.float32),
                    self.perspective_matrix
                )
                return tuple(pt[0][0])
            return (x, y)
        except Exception as e:
            logger.error(f"Error transforming point: {e}")
            return (x, y)


class TrafficAnalyzer:
    """
    Production-grade traffic analysis using YOLO + ByteTrack.
    Provides detection, tracking, speed estimation, and comprehensive analytics.
    """

    def __init__(self, model, fps: int = 30, calibration: Optional[CameraCalibration] = None):
        """
        Initialize Traffic Analyzer.
        
        Args:
            model: YOLO model instance
            fps: Video FPS
            calibration: CameraCalibration instance
        """
        self.model = model
        self.fps = fps
        self.calibration = calibration or CameraCalibration()

        # ByteTrack tracker
        try:
            import supervision as sv
            self.tracker = sv.ByteTrack(
                track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer=LOST_TRACK_BUFFER,
                minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
                frame_rate=fps,
                minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES
            )
        except ImportError:
            logger.error("supervision package not found. Install: pip install supervision")
            raise

        # State management
        self.track_history = defaultdict(lambda: deque(maxlen=SPEED_HISTORY_WINDOW))
        self.speed_history = defaultdict(lambda: deque(maxlen=SPEED_SMOOTHING_WINDOW))
        self.total_count = 0
        self.counted_ids = set()
        self.count_line_y = None

        # Vehicle classes
        self.vehicle_classes = VEHICLE_CLASSES

        # Annotators
        try:
            import supervision as sv
            self.box_annotator = sv.BoxAnnotator(thickness=BOX_THICKNESS)
            self.label_annotator = sv.LabelAnnotator(text_thickness=LABEL_TEXT_THICKNESS, text_scale=LABEL_TEXT_SCALE)
            self.trace_annotator = sv.TraceAnnotator(thickness=TRACE_THICKNESS, trace_length=TRACE_LENGTH)
        except ImportError as e:
            logger.error(f"Error importing annotators: {e}")
            raise

        # Analytics storage
        self.frame_analytics = []
        self.vehicle_database = {}
        logger.info("TrafficAnalyzer initialized successfully")

    def estimate_speed(self, track_id: int, current_pos: Tuple[float, float], frame_num: int) -> Optional[float]:
        """
        Estimate vehicle speed using Kalman-smoothed trajectory.
        
        Args:
            track_id: Tracker ID
            current_pos: (x, y) center position
            frame_num: Current frame number
            
        Returns:
            Speed in km/h or None if insufficient data
        """
        try:
            history = self.track_history[track_id]
            x, y = current_pos
            history.append((x, y, frame_num))

            if len(history) < 10:
                return None

            # Use frames 10 apart for stable estimate
            x1, y1, t1 = history[-10]
            x2, y2, t2 = history[-1]

            # Transform to bird's eye view for accurate distance
            if self.calibration and self.calibration.perspective_matrix is not None:
                p1 = self.calibration.transform_point(x1, y1)
                p2 = self.calibration.transform_point(x2, y2)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                pixel_dist = np.sqrt(dx**2 + dy**2)
            else:
                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Time in seconds
            time_diff = (t2 - t1) / self.fps

            if time_diff < SPEED_MIN_TIME_DIFF:
                return None

            # Convert to meters then km/h
            meters = self.calibration.pixel_to_meter(pixel_dist, y2) if self.calibration else pixel_dist * 0.04
            speed_ms = meters / time_diff
            speed_kmh = speed_ms * 3.6

            # Sanity check
            if speed_kmh > SPEED_MAX_THRESHOLD or speed_kmh < 0:
                return None

            # Smooth with history (median for robustness)
            self.speed_history[track_id].append(speed_kmh)
            smoothed_speed = np.median(self.speed_history[track_id])

            return smoothed_speed
        except Exception as e:
            logger.error(f"Error estimating speed for track {track_id}: {e}")
            return None

    def detect_direction(self, track_id: int, current_y: float) -> str:
        """
        Determine if vehicle is approaching or leaving.
        
        Args:
            track_id: Tracker ID
            current_y: Current Y position
            
        Returns:
            'approaching', 'leaving', or 'unknown'
        """
        try:
            history = self.track_history[track_id]
            if len(history) < 3:
                return 'unknown'

            recent_y = np.mean([h[1] for h in list(history)[-3:]])
            prev_y = np.mean([h[1] for h in list(history)[:3]])

            return 'approaching' if recent_y > prev_y else 'leaving'
        except Exception as e:
            logger.error(f"Error detecting direction for track {track_id}: {e}")
            return 'unknown'

    def count_vehicle(self, track_id: int, centre_y: float, direction: str) -> bool:
        """
        Count vehicle when it crosses the counting line.
        
        Args:
            track_id: Tracker ID
            centre_y: Center Y position
            direction: Direction ('approaching', 'leaving', 'unknown')
            
        Returns:
            True if vehicle was counted, False otherwise
        """
        try:
            if track_id in self.counted_ids or self.count_line_y is None:
                return False

            if direction == 'approaching':
                history = self.track_history[track_id]
                if len(history) >= 2:
                    prev_y = history[-2][1]
                    if prev_y < self.count_line_y <= centre_y:
                        self.total_count += 1
                        self.counted_ids.add(track_id)
                        self.vehicle_database[track_id] = {
                            'counted_at_frame': len(self.frame_analytics),
                            'direction': direction
                        }
                        logger.info(f"Vehicle {track_id} counted. Total: {self.total_count}")
                        return True

            return False
        except Exception as e:
            logger.error(f"Error counting vehicle {track_id}: {e}")
            return False

    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, Dict]:
        """
        Full pipeline: detect → track → analyze → annotate.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            
        Returns:
            Tuple of (annotated_frame, analytics_dict)
        """
        try:
            h, w = frame.shape[:2]
            if self.count_line_y is None:
                self.count_line_y = int(h * COUNT_LINE_Y_RATIO)

            # Detections
            results = self.model(frame, verbose=False)[0]
            import supervision as sv
            detections = sv.Detections.from_ultralytics(results)

            # Filter vehicles
            mask = np.isin(detections.class_id, list(self.vehicle_classes.keys()))
            detections = detections[mask]

            # Handle empty detections
            if len(detections) == 0:
                annotated = frame.copy()
                cv2.line(annotated, (0, self.count_line_y), (w, self.count_line_y), (0, 255, 255), 2)
                analytics = {
                    'frame': frame_num,
                    'timestamp': frame_num / self.fps,
                    'vehicle_count': self.total_count,
                    'active_tracks': 0,
                    'speeds': {},
                    'directions': {}
                }
                self.frame_analytics.append(analytics)
                return annotated, analytics

            # Track vehicles
            detections = self.tracker.update_with_detections(detections)

            # Analytics
            labels = []
            speeds = {}
            directions = {}

            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                conf = detections.confidence[i]
                cls_id = detections.class_id[i]
                track_id = detections.tracker_id[i]

                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Speed estimation
                speed = self.estimate_speed(track_id, (cx, cy), frame_num)
                speeds[track_id] = speed

                # Direction detection
                direction = self.detect_direction(track_id, cy)
                directions[track_id] = direction

                # Count vehicle
                self.count_vehicle(track_id, cy, direction)

                # Label
                class_name = self.vehicle_classes.get(int(cls_id), 'vehicle')
                if speed:
                    label = f"#{track_id} {class_name} {speed:.1f}km/h"
                else:
                    label = f"#{track_id} {class_name} ..."
                labels.append(label)

            # Annotations
            annotated = frame.copy()

            # Counting line
            cv2.line(annotated, (0, self.count_line_y), (w, self.count_line_y), (0, 255, 255), 2)
            cv2.putText(annotated, "COUNT LINE", (10, self.count_line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Boxes, labels, traces
            annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)

            # HUD
            hud_bg = annotated.copy()
            cv2.rectangle(hud_bg, (0, 0), (350, 120), (0, 0, 0), -1)
            annotated = cv2.addWeighted(annotated, 1, hud_bg, HUD_BG_ALPHA, 0)

            cv2.putText(annotated, f"TOTAL COUNT: {self.total_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"ACTIVE: {len(detections)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {self.fps}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Store analytics
            analytics = {
                'frame': frame_num,
                'timestamp': frame_num / self.fps,
                'vehicle_count': self.total_count,
                'active_tracks': len(detections),
                'speeds': {int(k): round(v, 2) if v else None for k, v in speeds.items()},
                'directions': {int(k): v for k, v in directions.items()}
            }

            self.frame_analytics.append(analytics)

            return annotated, analytics
        except Exception as e:
            logger.error(f"Error processing frame {frame_num}: {e}")
            raise

    def export_analytics(self, output_path: str) -> None:
        """
        Export analytics to JSON file.
        
        Args:
            output_path: Path to save analytics JSON
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.frame_analytics, f, indent=2)
            logger.info(f"Analytics exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")


def process_video(video_path: str, output_path: str, model_name: str = MODEL_NAME) -> Dict:
    """
    Process video file with traffic detection and tracking.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model_name: YOLO model name
        
    Returns:
        Dictionary with processing statistics
    """
    try:
        from ultralytics import YOLO
        import time

        logger.info(f"Starting video processing: {video_path}")

        # Load model
        model = YOLO(model_name)
        logger.info(f"Model {model_name} loaded")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")

        # Initialize calibration and analyzer
        calibration = CameraCalibration(pixel_to_meter_ratio=PIXEL_TO_METER_RATIO)
        calibration.auto_init_from_frame(w, h)
        analyzer = TrafficAnalyzer(model, fps=int(fps), calibration=calibration)

        # Setup output
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC), fps, (w, h))
        if not writer.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")

        logger.info(f"Processing {total_frames} frames...")
        start_time = time.time()

        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _ = analyzer.process_frame(frame, frame_num)
            writer.write(annotated)

            if VERBOSE and frame_num % 30 == 0:
                elapsed = time.time() - start_time
                speed = (frame_num + 1) / elapsed
                logger.info(f"Frame {frame_num}/{total_frames} | {speed:.1f} FPS | Count: {analyzer.total_count}")

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Complete!")
        logger.info(f"   Frames: {total_frames}")
        logger.info(f"   Time: {elapsed:.1f}s")
        logger.info(f"   Speed: {total_frames/elapsed:.1f} FPS")
        logger.info(f"   Vehicles Counted: {analyzer.total_count}")
        logger.info(f"   Unique Tracks: {len(analyzer.vehicle_database)}")
        logger.info(f"{'='*50}")

        # Export analytics
        if SAVE_ANALYTICS:
            analytics_path = output_path.replace('.mp4', '_analytics.json')
            analyzer.export_analytics(analytics_path)

        return {
            'total_frames': total_frames,
            'processing_time': elapsed,
            'fps': total_frames / elapsed,
            'vehicles_counted': analyzer.total_count,
            'unique_tracks': len(analyzer.vehicle_database),
            'output_video': output_path,
            'analytics_file': analytics_path if SAVE_ANALYTICS else None
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python traffic_analyzer.py <input_video> <output_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    stats = process_video(input_video, output_video)
    print(f"\nProcessing complete! Stats: {stats}")
