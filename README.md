# Traffic Violation Detection System

A comprehensive AI-powered system for detecting traffic violations in vehicles using computer vision and deep learning. This system can analyze images of cars and bikes to detect various violations including seatbelt usage, helmet compliance, mobile phone usage, and more.

## 🚀 Features

### Car Detection
- **Seatbelt Violation Detection**: Detects drivers and passengers not wearing seatbelts
- **Number Plate Recognition**: Automatic Number Plate Recognition (ANPR) using PARSeq and CRNN models
- **Dual Model Architecture**: Uses two YOLO models for improved accuracy
- **Adaptive Detection**: Automatically adjusts for day/night conditions and aerial/side views

### Bike Detection
- **Helmet Violation Detection**: Detects riders and pillion riders not wearing helmets
- **Mobile Usage Detection**: Identifies mobile phone usage while riding
- **Triple Riding Detection**: Detects when three or more people are on a bike
- **Number Plate Recognition**: ANPR for bike registration plates

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](./API_DOCUMENTATION.md)
- [Setup Guide](./SETUP.md)
- [Architecture](./ARCHITECTURE.md)
- [Models](./MODELS.md)
- [Usage Examples](./USAGE.md)

## 🏗️ Project Structure

```
traffic_violation/
├── main.py                 # FastAPI application with REST endpoints
├── requirements.txt        # Python dependencies
├── models/                 # YOLO model files
│   ├── car.pt             # Main car detection model
│   ├── car_yolov11.pt     # Seatbelt detection model
│   └── bike.pt            # Bike violation detection model
├── src/                   # Source code
│   ├── detector.py        # Car detection logic
│   ├── detector2.py       # Bike detection logic
│   ├── model.py          # CRNN model definition
│   └── modules/          # Neural network modules
├── parseq/               # PARSeq text recognition library
├── folders/              # Output folders for car detections
├── bike_folders/         # Output folders for bike detections
└── docs/                 # Documentation
```

## 🚦 Quick Start

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PARSeq** (if using local parseq directory):
   ```bash
   pip install -e ./parseq/parseq
   ```

4. **Start the API server**:
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API**:
   - API will be available at `http://localhost:8000`
   - Interactive API docs at `http://localhost:8000/docs`
   - Alternative docs at `http://localhost:8000/redoc`

## 📡 API Endpoints

### Car Detection
- **POST** `/analyze-car/` - Analyze car images for violations

### Bike Detection
- **POST** `/analyze-bike/` - Analyze bike images for violations

### Information
- **GET** `/` - API information and available endpoints

For detailed API documentation, see [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

## 🧠 Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework
- **Ultralytics YOLO**: Object detection models
- **OpenCV**: Image processing
- **PARSeq**: Scene text recognition
- **Pillow**: Image manipulation
- **NumPy**: Numerical computing

## 📊 Detection Capabilities

### Car Violations
- Driver not wearing seatbelt
- Passenger not wearing seatbelt
- Number plate detection and OCR

### Bike Violations
- Rider not wearing helmet
- Pillion rider not wearing helmet
- Rider and pillion both not wearing helmet
- Mobile phone usage while riding
- Triple riding (3+ people on bike)
- Number plate detection and OCR

## 🔧 Configuration

Model paths and other configurations can be found in:
- `src/detector.py` - Car detection configuration
- `src/detector2.py` - Bike detection configuration

