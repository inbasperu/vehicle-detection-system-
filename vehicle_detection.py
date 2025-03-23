import argparse
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path=None, confidence=0.3, device="cpu"):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to the YOLOv8 model (if None, will download the model)
            confidence: Detection confidence threshold
            device: Device to run the model on ('cpu', 'cuda', 'mps')
        """
        # Load YOLOv8 model
        self.model = YOLO("yolov8n.pt" if model_path is None else model_path)
        self.confidence = confidence
        self.device = device
        
        # Define vehicle classes from COCO dataset
        self.target_classes = [
            1,  # bicycle
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
        ]
        
        # Class names for display
        self.class_names = self.model.names
        
        # Initialize counters
        self.total_vehicles_crossed = 0
        self.previous_vehicle_count = 0
        self.current_centers = {}  # Store centers of tracked vehicles
        self.counted_ids = set()  # Store IDs of vehicles that have been counted
        
        # Line for counting (will be initialized during setup)
        self.line_y = None
        
    def setup_counter(self, frame):
        """Set up the counting line based on the first frame dimensions."""
        frame_height, frame_width = frame.shape[:2]
        # Create a counting line in the middle of the frame
        self.line_y = frame_height // 2
        self.frame_width = frame_width
        self.frame_height = frame_height

    def process_frame(self, frame):
        """
        Process a frame to detect and count vehicles.
        
        Args:
            frame: Input image frame
            
        Returns:
            annotated_frame: Frame with annotations
            vehicle_count: Number of vehicles in the frame
            total_vehicles_crossed: Number of vehicles that crossed the line
        """
        # Initialize counter if not already done
        if self.line_y is None:
            self.setup_counter(frame)
            
        # Format frame for YOLOv8
        results = self.model.track(
            frame, 
            conf=self.confidence, 
            classes=self.target_classes,
            device=self.device,
            persist=True,  # Enable tracking
            verbose=False
        )[0]
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw the counting line
        cv2.line(
            annotated_frame,
            (0, self.line_y),
            (self.frame_width, self.line_y),
            (255, 0, 0),  # Red color
            2
        )
        
        # Get current detections
        boxes = results.boxes.cpu().numpy()
        vehicle_count = len(boxes)
        
        # Process each detected vehicle
        for box in boxes:
            # Get detection info
            xyxy = box.xyxy[0]  # Get bounding box coordinates (x1, y1, x2, y2)
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            track_id = int(box.id[0]) if box.id is not None else -1  # Tracking ID

            # If tracking is enabled and we have a valid ID
            if track_id != -1:
                # Calculate center of the bounding box
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                
                # Store current center position
                if track_id in self.current_centers:
                    prev_y = self.current_centers[track_id]
                    
                    # Check if vehicle crossed the line (from top to bottom)
                    if prev_y < self.line_y and y_center >= self.line_y:
                        if track_id not in self.counted_ids:
                            self.total_vehicles_crossed += 1
                            self.counted_ids.add(track_id)
                    
                    # Check if vehicle crossed the line (from bottom to top)
                    elif prev_y >= self.line_y and y_center < self.line_y:
                        if track_id not in self.counted_ids:
                            self.total_vehicles_crossed += 1
                            self.counted_ids.add(track_id)
                
                # Update center position
                self.current_centers[track_id] = y_center
                
                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (0, 255, 0),  # Green color
                    2
                )
                
                # Add label with class name, confidence, and tracking ID
                label = f"{self.class_names[cls]} {conf:.2f} #{track_id}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),  # Green color
                    2
                )
        
        # Add line counter text
        line_counter_text = f"Vehicles crossed line: {self.total_vehicles_crossed}"
        cv2.putText(
            annotated_frame,
            line_counter_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),  # Red color
            2
        )
        
        # Add current vehicle count text
        current_count_text = f"Current vehicles: {vehicle_count}"
        cv2.putText(
            annotated_frame,
            current_count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Blue color
            2
        )
        
        return (
            annotated_frame, 
            vehicle_count,
            self.total_vehicles_crossed
        )


def process_video(input_source, output_path=None, model_path=None, confidence=0.3, device="cpu", display=True):
    """
    Process a video file or camera stream for vehicle detection and counting.
    
    Args:
        input_source: Path to video file or camera index (0 for default camera)
        output_path: Path to save the output video (None for no save)
        model_path: Path to YOLOv8 model
        confidence: Detection confidence threshold
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        display: Whether to display the output video
    """
    # Initialize the detector
    detector = VehicleDetector(model_path, confidence, device)
    
    # Open video capture
    if isinstance(input_source, int) or input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
        print(f"Starting camera stream from camera index {input_source}")
    else:
        cap = cv2.VideoCapture(input_source)
        print(f"Processing video file: {input_source}")
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),  # Change codec as needed
            fps,
            (frame_width, frame_height)
        )
    
    # Process video frames
    frame_count = 0
    total_fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Measure processing time
            start_time = time.time()
            
            # Process frame
            annotated_frame, current_count, total_count = detector.process_frame(frame)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            # Add FPS to the frame
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green color
                2
            )
            
            # Write frame to output video
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            if display:
                cv2.imshow("Vehicle Detection", annotated_frame)
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print statistics (optional)
            print(f"\rFrame: {frame_count}, Current vehicles: {current_count}, Total crossed: {total_count}, FPS: {fps:.1f}", end="")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Print final statistics
    avg_fps = total_fps / frame_count if frame_count > 0 else 0
    print(f"\nProcessed {frame_count} frames at an average of {avg_fps:.1f} FPS")
    print(f"Total vehicles that crossed the line: {detector.total_vehicles_crossed}")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    """Parse command line arguments and run the vehicle detector."""
    parser = argparse.ArgumentParser(description="Vehicle Detection and Counting System")
    parser.add_argument(
        "--source", 
        default="0", 
        help="Path to video file or camera index (default: 0 for webcam)"
    )
    parser.add_argument(
        "--output", 
        default=None, 
        help="Path to save output video (default: None)"
    )
    parser.add_argument(
        "--model", 
        default=None, 
        help="Path to YOLOv8 model (default: None, will use yolov8n.pt)"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.3, 
        help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--device", 
        default="cpu", 
        help="Device to run inference (cpu, cuda, mps) (default: cpu)"
    )
    parser.add_argument(
        "--no-display", 
        action="store_true", 
        help="Don't display video during processing"
    )
    
    args = parser.parse_args()
    
    # Run vehicle detection
    process_video(
        input_source=args.source,
        output_path=args.output,
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()