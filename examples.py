"""
Example usage of Traffic Detection & Tracking System
"""

import cv2
from ultralytics import YOLO
from traffic_analyzer import TrafficAnalyzer, CameraCalibration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """
    Basic example: Process a video file
    """
    logger.info("Running basic usage example...")
    
    from traffic_analyzer import process_video
    
    # Process video
    stats = process_video(
        video_path="input_video.mp4",
        output_path="output_tracked.mp4"
    )
    
    print("\n✅ Processing Complete!")
    print(f"Vehicles Counted: {stats['vehicles_counted']}")
    print(f"Processing Speed: {stats['fps']:.1f} FPS")
    print(f"Output Video: {stats['output_video']}")


def example_custom_configuration():
    """
    Advanced example: Custom configuration and real-time processing
    """
    logger.info("Running custom configuration example...")
    
    # Load model
    model = YOLO("yolov8s.pt")
    
    # Custom calibration
    calibration = CameraCalibration(pixel_to_meter_ratio=0.035)
    calibration.auto_init_from_frame(1920, 1080)
    
    # Create analyzer with custom parameters
    analyzer = TrafficAnalyzer(model, fps=30, calibration=calibration)
    
    # Open video
    cap = cv2.VideoCapture("input_video.mp4")
    writer = cv2.VideoWriter(
        "output_custom.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (1920, 1080)
    )
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, analytics = analyzer.process_frame(frame, frame_count)
            writer.write(annotated)
            
            # Display stats every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | Count: {analyzer.total_count} | Active: {analytics['active_tracks']}")
            
            frame_count += 1
    
    finally:
        cap.release()
        writer.release()
        
        # Export analytics
        analyzer.export_analytics("output_analytics.json")
        print(f"\n✅ Processing complete! Total vehicles: {analyzer.total_count}")


def example_stream_processing():
    """
    Example: Real-time camera stream processing
    """
    logger.info("Running stream processing example...")
    
    model = YOLO("yolov8s.pt")
    calibration = CameraCalibration(pixel_to_meter_ratio=0.035)
    
    # Use webcam (0) or IP camera URL
    cap = cv2.VideoCapture(0)  # Change to camera URL for IP camera
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    calibration.auto_init_from_frame(w, h)
    analyzer = TrafficAnalyzer(model, fps=int(fps), calibration=calibration)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, analytics = analyzer.process_frame(frame, 0)
            cv2.imshow("Traffic Detection", annotated)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stream processing stopped. Total vehicles: {analyzer.total_count}")


def example_batch_processing():
    """
    Example: Process multiple videos in batch
    """
    logger.info("Running batch processing example...")
    
    import os
    import glob
    from traffic_analyzer import process_video
    
    # Process all MP4 files in a directory
    video_dir = "videos"
    output_dir = "output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in glob.glob(os.path.join(video_dir, "*.mp4")):
        output_file = os.path.join(output_dir, os.path.basename(video_file))
        
        print(f"\nProcessing: {video_file}")
        try:
            stats = process_video(video_file, output_file)
            print(f"✅ Complete | Vehicles: {stats['vehicles_counted']} | Speed: {stats['fps']:.1f} FPS")
        except Exception as e:
            print(f"❌ Error: {e}")


def example_custom_counting_line():
    """
    Example: Set custom counting line position
    """
    logger.info("Running custom counting line example...")
    
    from traffic_analyzer import TrafficAnalyzer
    from ultralytics import YOLO
    import numpy as np
    
    model = YOLO("yolov8s.pt")
    analyzer = TrafficAnalyzer(model, fps=30)
    
    # Process a frame to initialize
    cap = cv2.VideoCapture("input_video.mp4")
    ret, frame = cap.read()
    
    if ret:
        h, w = frame.shape[:2]
        
        # Set counting line at 70% from top instead of default 60%
        analyzer.count_line_y = int(h * 0.7)
        
        print(f"Counting line set at y={analyzer.count_line_y} (70% from top)")
    
    cap.release()


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Basic Usage", example_basic_usage),
        "2": ("Custom Configuration", example_custom_configuration),
        "3": ("Stream Processing", example_stream_processing),
        "4": ("Batch Processing", example_batch_processing),
        "5": ("Custom Counting Line", example_custom_counting_line),
    }
    
    print("\n🚗 Traffic Detection & Tracking - Examples")
    print("=" * 50)
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Select example (1-5): ")
    
    if choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")
