import argparse
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path=None, confidence=0.3, device="cpu", 
                 roi_x1_percent=20, roi_y1_percent=20, roi_x2_percent=80, roi_y2_percent=80): 
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to the YOLOv8 model (if None, will download the default model)
            confidence: Detection confidence threshold
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            roi_x1_percent: Top-left X coordinate of ROI (% of width).
            roi_y1_percent: Top-left Y coordinate of ROI (% of height).
            roi_x2_percent: Bottom-right X coordinate of ROI (% of width).
            roi_y2_percent: Bottom-right Y coordinate of ROI (% of height).
        """
        # Load YOLOv8 model - Defaulting to yolov8x.pt for higher accuracy
        self.model = YOLO("yolov8x.pt" if model_path is None else model_path)
        self.confidence = confidence
        self.device = device
        # Store ROI percentages
        self.roi_x1_percent = roi_x1_percent
        self.roi_y1_percent = roi_y1_percent
        self.roi_x2_percent = roi_x2_percent
        self.roi_y2_percent = roi_y2_percent
        
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
        self.class_counts = {cls_id: 0 for cls_id in self.target_classes} # Counts per class
        self.counted_ids = set()  # Store IDs of vehicles that have been counted
        
        # Frame dimensions and ROI pixel coordinates (will be initialized)
        self.frame_height = None
        self.frame_width = None
        self.roi_x1_px = None
        self.roi_y1_px = None
        self.roi_x2_px = None
        self.roi_y2_px = None

    def _initialize_frame_params(self, frame):
        """Initialize frame parameters and calculate ROI pixel coordinates."""
        if self.frame_height is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # Calculate ROI pixel coordinates from percentages
            self.roi_x1_px = int(self.frame_width * (self.roi_x1_percent / 100.0))
            self.roi_y1_px = int(self.frame_height * (self.roi_y1_percent / 100.0))
            self.roi_x2_px = int(self.frame_width * (self.roi_x2_percent / 100.0))
            self.roi_y2_px = int(self.frame_height * (self.roi_y2_percent / 100.0))

            # Basic validation
            if not (0 <= self.roi_x1_px < self.roi_x2_px <= self.frame_width and \
                    0 <= self.roi_y1_px < self.roi_y2_px <= self.frame_height):
                print("Error: Invalid ROI coordinates calculated. Disabling ROI counting.")
                # Set invalid coordinates to prevent processing
                self.roi_x1_px = self.roi_y1_px = self.roi_x2_px = self.roi_y2_px = -1
                
            print(f"Detector using frame dimensions: {self.frame_width}x{self.frame_height}.")
            print(f"ROI defined: TopLeft=({self.roi_x1_px},{self.roi_y1_px}), BottomRight=({self.roi_x2_px},{self.roi_y2_px})")


    def process_frame(self, frame):
        """
        Process a frame to detect and count vehicles entering the defined ROI.
        
        Args:
            frame: Input image frame (potentially downsampled)
            
        Returns:
            annotated_frame: Frame (potentially downsampled) with annotations
            vehicle_count: Number of vehicles currently in the frame
            class_counts: Dictionary of counts per vehicle class that entered the ROI
        """
        self._initialize_frame_params(frame)
            
        results = self.model.track(
            frame, conf=self.confidence, classes=self.target_classes,
            device=self.device, persist=True, verbose=False
        )[0]
        
        annotated_frame = frame.copy()

        # --- Draw ROI Rectangle ---
        if self.roi_x1_px is not None and self.roi_x1_px >= 0: # Check if ROI is valid
             cv2.rectangle(
                 annotated_frame,
                 (self.roi_x1_px, self.roi_y1_px),
                 (self.roi_x2_px, self.roi_y2_px),
                 (0, 255, 255),  # Yellow color for ROI
                 2 # Line thickness
             )

        boxes = results.boxes.cpu().numpy()
        vehicle_count = len(boxes)
        
        for box in boxes:
            xyxy = box.xyxy[0] 
            conf = box.conf[0] 
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if track_id != -1:
                # Calculate center of the bounding box
                x_center = (xyxy[0] + xyxy[2]) / 2 
                y_center = (xyxy[1] + xyxy[3]) / 2
                
                # --- Updated Counting Logic: Check if center is inside ROI ---
                is_inside_roi = False
                if self.roi_x1_px is not None and self.roi_x1_px >= 0: # Check if ROI is valid
                    is_inside_roi = (self.roi_x1_px < x_center < self.roi_x2_px) and \
                                    (self.roi_y1_px < y_center < self.roi_y2_px)

                if is_inside_roi and track_id not in self.counted_ids:
                     if cls in self.class_counts:
                         self.class_counts[cls] += 1
                         self.counted_ids.add(track_id)
                         print(f"\nEntered ROI: ID {track_id}, Class: {self.class_names[cls]}, New Count: {self.class_counts[cls]}") 
                
                # --- Drawing Logic ---
                cv2.rectangle(
                    annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                    (0, 255, 0), 2
                )
                
                label = f"{self.class_names[cls]} {conf:.2f} #{track_id}"
                # Highlight counted vehicles differently (e.g., red label)
                label_color = (0, 0, 255) if track_id in self.counted_ids else (0, 255, 0) 
                cv2.putText(
                    annotated_frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2
                )
        
        # --- Text Display ---
        current_count_text = f"Current vehicles: {vehicle_count}"
        cv2.putText(annotated_frame, current_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        text_y_offset = 60 
        for cls_id, count in self.class_counts.items():
             if cls_id in self.class_names: 
                class_name = self.class_names[cls_id]
                # Updated text description
                class_count_text = f"{class_name.capitalize()}s Entered ROI: {count}" 
                cv2.putText(annotated_frame, class_count_text, (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                text_y_offset += 30 

        return (annotated_frame, vehicle_count, self.class_counts)


def process_video(input_source, output_path=None, model_path=None, confidence=0.3, 
                  device="cpu", display=True, downsample_factor=1, 
                  roi_x1=20, roi_y1=20, roi_x2=80, roi_y2=80,
                  process_interval_seconds=0.0):
    """
    Process a video file or camera stream for vehicle detection and counting within an ROI.
    
    Args:
        input_source: Path to video file or camera index (0 for default camera)
        output_path: Path to save the output video (None for no save)
        model_path: Path to YOLOv8 model
        confidence: Detection confidence threshold
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        display: Whether to display the output video
        downsample_factor: Factor to downsample the frame before processing (e.g., 2 means half width/height)
        roi_x1, roi_y1, roi_x2, roi_y2: ROI corner coordinates as percentages (0-100).
        process_interval_seconds: Process one frame every N seconds (0.0 means process every frame)
    """
    # Input validation
    if not isinstance(downsample_factor, int) or downsample_factor < 1:
        print("Warning: Invalid downsample_factor. Using 1.")
        downsample_factor = 1
    if not (0 <= roi_x1 < roi_x2 <= 100 and 0 <= roi_y1 < roi_y2 <= 100):
        print("Error: Invalid ROI percentage values (must be 0-100, x1<x2, y1<y2). Check arguments.")
        return
    if process_interval_seconds < 0:
        print("Warning: Invalid process_interval_seconds. Using 0.0 (process every frame).")
        process_interval_seconds = 0.0

    # Initialize the detector, passing the ROI percentages
    detector = VehicleDetector(model_path, confidence, device, roi_x1, roi_y1, roi_x2, roi_y2)
    
    cap = cv2.VideoCapture(int(input_source) if isinstance(input_source, int) or input_source.isdigit() else input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) 
    
    print(f"Original frame size: {original_width}x{original_height}")
    print(f"Original FPS: {fps_video:.2f}")
    if downsample_factor > 1: print(f"Downsampling factor: {downsample_factor}")
    if process_interval_seconds > 0: 
        print(f"Processing one frame every {process_interval_seconds} seconds")
        print(f"Expected compute savings: {1.0 - (1.0 / (fps_video * process_interval_seconds)):.1%}")
        
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (original_width, original_height))
    
    frame_count = 0
    total_fps = 0
    last_process_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Skip frames based on process_interval_seconds
            if process_interval_seconds > 0 and (current_time - last_process_time) < process_interval_seconds:
                # Skip reading the frame entirely to save compute power
                cap.grab()  # This is much faster than cap.read()
                continue
            
            last_process_time = current_time
            
            # Read and process the frame
            ret, original_frame = cap.read()
            if not ret: break
            
            # Downsampling
            if downsample_factor > 1:
                process_width = max(1, original_width // downsample_factor)
                process_height = max(1, original_height // downsample_factor)
                input_frame = cv2.resize(original_frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
            else:
                input_frame = original_frame 

            start_time = time.time()
            annotated_processed_frame, current_count, class_counts = detector.process_frame(input_frame)
            end_time = time.time()
            
            current_fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            total_fps += current_fps
            frame_count += 1

            # Prepare display frame (upscale if needed)
            if downsample_factor > 1:
                 display_frame = cv2.resize(annotated_processed_frame, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            else:
                 display_frame = annotated_processed_frame 

            # Add FPS text
            fps_text_y = 120 
            if detector.frame_height is not None: 
                 scale_factor = original_height / detector.frame_height if detector.frame_height else 1
                 base_offset_scaled = int(60 * scale_factor)
                 line_height_scaled = int(30 * scale_factor)
                 # Add space if ROI text is also scaled later
                 fps_text_y = base_offset_scaled + (len(detector.class_counts) * line_height_scaled) + int(10 * scale_factor) # Add some padding

            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, fps_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if writer: writer.write(display_frame)
            if display:
                cv2.imshow("Vehicle Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Print statistics
            class_counts_str = ", ".join([f"{detector.class_names[k].capitalize()}: {v}" for k, v in class_counts.items()])
            print(f"\rFrame: {frame_count}, Current: {current_count}, Entered ROI: [{class_counts_str}], FPS: {current_fps:.1f}   ", end="")

    except KeyboardInterrupt: print("\nInterrupted by user")
    
    avg_fps = total_fps / frame_count if frame_count > 0 else 0
    print(f"\nProcessed {frame_count} frames at an average of {avg_fps:.1f} FPS")
    
    final_counts_str = "\n".join([f"  - {detector.class_names[k].capitalize()}: {v}" for k, v in detector.class_counts.items()])
    print(f"Total vehicles counted entering ROI ({roi_x1}%,{roi_y1}% to {roi_x2}%,{roi_y2}%):") 
    print(final_counts_str)
    
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


def main():
    """Parse command line arguments and run the vehicle detector."""
    parser = argparse.ArgumentParser(description="Vehicle Detection and Counting System using ROI")
    parser.add_argument("--source", default="0", help="Video file or camera index (default: 0)")
    parser.add_argument("--output", default=None, help="Path to save output video (default: None)")
    parser.add_argument("--model", default=None, help="Path to YOLOv8 model (default: yolov8x.pt)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps) (default: cpu)")
    parser.add_argument("--no-display", action="store_true", help="Don't display video")
    parser.add_argument("--downsample-factor", type=int, default=1, help="Downsample factor (default: 1)")
    parser.add_argument("--process-interval", type=float, default=0.0, help="Process one frame every N seconds (default: 0.0, process every frame)")
    
    # --- ROI Arguments ---
    parser.add_argument("--roi-x1", type=float, default=20.0, help="ROI top-left X (% width, 0-100, default: 20)")
    parser.add_argument("--roi-y1", type=float, default=20.0, help="ROI top-left Y (% height, 0-100, default: 20)")
    parser.add_argument("--roi-x2", type=float, default=80.0, help="ROI bottom-right X (% width, 0-100, default: 80)")
    parser.add_argument("--roi-y2", type=float, default=80.0, help="ROI bottom-right Y (% height, 0-100, default: 80)")

    args = parser.parse_args()
    
    # Pass ROI args to process_video
    process_video(
        input_source=args.source, output_path=args.output, model_path=args.model,
        confidence=args.confidence, device=args.device, display=not args.no_display,
        downsample_factor=args.downsample_factor,
        roi_x1=args.roi_x1, roi_y1=args.roi_y1, roi_x2=args.roi_x2, roi_y2=args.roi_y2,
        process_interval_seconds=args.process_interval
    )

if __name__ == "__main__":
    main()