# Vehicle Detection System - Installation & Usage Guide

This guide will help you set up and run the Python-based vehicle detection system using YOLOv8.

## Features

- Real-time vehicle detection and counting
- Support for both webcam feeds and video files
- Line crossing detection to count vehicles passing by
- Vehicle presence detection in the frame
- Performance metrics (FPS)
- Save processed video with annotations

## Installation

1. **Clone or download the code files**

2. **Set up Python environment** (recommended Python 3.8+)

   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the detection system with default settings (using webcam):

```bash
python vehicle_detection.py
```

### Command Line Options

- `--source`: Specify the input source (webcam index or video file path)

  ```bash
  # Use webcam (default)
  python vehicle_detection.py --source 0
  
  # Use a video file
  python vehicle_detection.py --source path/to/your/video.mp4
  ```

- `--output`: Save the processed video with annotations

  ```bash
  python vehicle_detection.py --source path/to/your/video.mp4 --output output_video.mp4
  ```

- `--model`: Specify a custom YOLOv8 model file (if you have one)

  ```bash
  python vehicle_detection.py --model path/to/custom_model.pt
  ```

- `--confidence`: Set the detection confidence threshold (default: 0.3)

  ```bash
  python vehicle_detection.py --confidence 0.5
  ```

- `--device`: Choose the processing device (default: cpu)

  ```bash
  # Use CPU (default)
  python vehicle_detection.py --device cpu
  
  # Use GPU with CUDA (if available)
  python vehicle_detection.py --device cuda
  
  # Use Apple M1/M2 GPU (if available)
  python vehicle_detection.py --device mps
  ```

- `--no-display`: Run without displaying the video (useful for headless servers)

  ```bash
  python vehicle_detection.py --source video.mp4 --output processed.mp4 --no-display
  ```

### Complete Example

Process a video file, save the output, use GPU acceleration, and set a higher confidence threshold:

```bash
python vehicle_detection.py --source traffic_video.mp4 --output processed_traffic.mp4 --device cuda --confidence 0.4
```

## Controls

- Press 'q' to quit the application while it's running

## Understanding the Output

The processed video will show:

- Bounding boxes around detected vehicles with labels (type, confidence, tracking ID)
- A horizontal line in the middle used for counting vehicles passing by
- Total count of vehicles that have crossed the line
- Current number of vehicles in the frame
- Real-time processing speed (FPS)

## Troubleshooting

1. **Low FPS or slow performance**
   - Try running on a GPU if available (`--device cuda`)
   - Lower the resolution of your input video
   - Use a lighter YOLOv8 model (yolov8n.pt is used by default)

2. **Missing dependencies**
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - For CUDA support, ensure you have compatible NVIDIA drivers and CUDA installed

3. **Camera not working**
   - Try different camera indices: `--source 1`, `--source 2`, etc.
   - Check if your camera works with other applications

4. **Model download issues**
   - If the automatic model download fails, manually download the YOLOv8 model from the Ultralytics website and specify its path with `--model`
