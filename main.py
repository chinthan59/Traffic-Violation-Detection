import sys
import os

# Add parseq directory to Python path for strhub import
parseq_path = os.path.join(os.path.dirname(__file__), 'parseq')
if parseq_path not in sys.path:
    sys.path.insert(0, parseq_path)

from fastapi import FastAPI, UploadFile, File, Response
import shutil
import re
import tempfile
import io
from PIL import Image
from src.detector import process_car_image
from src.detector_new import process_car_image as process_car_image_new
from src.detector2 import process_bike_image, process_bike_image_with_annotation
from src.detector2_bike_new import process_bike_image as process_bike_image_new

app = FastAPI()


@app.post("/analyze-car/")
async def analyze_car(file: UploadFile = File(...)):
    """
    API endpoint for car detection and violation analysis.
    
    Takes an image as input and returns:
    - Number plate detection and OCR
    - Seatbelt violations (driver/passenger not wearing seatbelt)
    - Bounding boxes for plates and violations
    
    Returns JSON response with detections and violations.
    """
    safe_filename = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{safe_filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = process_car_image(temp_path, safe_filename)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze-bike/")
async def analyze_bike(file: UploadFile = File(...)):
    """
    API endpoint for bike detection and violation analysis.
    
    Takes an image as input and returns:
    - Number plate detection and OCR
    - Bike violations (helmet violations, mobile usage, triple riding, etc.)
    - Bounding boxes for plates and violations
    
    Returns JSON response with detections and violations.
    """
    safe_filename = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{safe_filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = process_bike_image(temp_path, safe_filename)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze-car-new/")
async def analyze_car_new(file: UploadFile = File(...)):
    """
    API endpoint for enhanced car detection and violation analysis.
    
    Uses dual-model detection (best.pt + car_yolov11_new.pt) with:
    - Advanced 2-line number plate recognition
    - No_passenger filtering for false positive reduction
    - High-precision seatbelt violation detection
    - ANPR confidence scores
    
    Takes an image as input and returns:
    - Number plate detection and OCR with confidence
    - Seatbelt violations (driver/passenger not wearing seatbelt)
    - Bounding boxes for plates and violations
    
    Returns JSON response with detections and violations.
    """
    safe_filename = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{safe_filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = process_car_image_new(temp_path, safe_filename)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/analyze-bike-new/")
async def analyze_bike_new(file: UploadFile = File(...)):
    """
    YOLOv11 Bike Violation Detector V2 (2026-02-10)
    
    Uses bike_yolov11_new.pt for enhanced bike violation detection with updated class structure.
    
    FEATURES:
      - Single YOLOv11 model (bike_yolov11_new.pt)
      - Automatic day/night detection
      - Adaptive confidence thresholds (0.25 for validation mode)
      - Helmet violation detection (rider and pillion separately)
      - Mobile usage detection
      - Triple riding detection
      - Wearing helmet verification to filter false positives
      - Spatial overlap checking with confidence comparison
      - Vehicle-based spatial grouping for aerial views
      - Deduplication using IoU and containment checks
      - PARSeq OCR with 2-line plate support
    
    MODEL CLASSES (bike_yolov11_new.pt):
      Class 0: 6 (IGNORED)
      Class 1: Number_plate
      Class 2: mobile_usage
      Class 3: pillion_not_wearing_helmet
      Class 4: pillion_wearing_helmet
      Class 5: rider_not_wearing_helmet
      Class 6: rider_wearing_helmet
      Class 7: triple_riding
      Class 8: vehicle
    
    DETECTION STRATEGY:
      - VALIDATION MODE: Uses low confidence thresholds (0.25) to see all detections
      - Resolves conflicting detections (wearing vs not wearing)
      - Groups detections by vehicle for accurate assignment
      - Deduplicates overlapping detections
      - Returns both violations and compliance detections
    
    VIOLATION TYPES DETECTED:
      - Rider not wearing helmet
      - Pillion not wearing helmet
      - Mobile usage while riding
      - Triple riding (3+ people on bike)
    
    RESPONSE FORMAT:
      {
        "status": "success",
        "timestamp": "2026-02-10T...",
        "image_id": "BIKE_...",
        "detections": [
          {
            "number_plate": {
              "plate_number": "KA01AB1234",
              "plate_bbox": [x1, y1, x2, y2]
            },
            "violations": [
              {
                "type": "rider_not_wearing_helmet",
                "confidence": 0.85,
                "bbox": [x1, y1, x2, y2]
              }
            ]
          }
        ]
      }
    """
    safe_filename = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_{safe_filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = process_bike_image_new(temp_path, safe_filename)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Traffic Violation Detection API",
        "endpoints": {
            "/analyze-car/": "POST - Analyze car images for seatbelt violations (standard detector)",
            "/analyze-car-new/": "POST - Analyze car images with enhanced dual-model detection (best.pt + car_yolov11_new.pt)",
            "/analyze-bike/": "POST - Analyze bike images for helmet and other violations",
            "/analyze-bike-annotated/": "POST - Analyze bike images and return annotated image (uses both bike_old.pt and bike.pt models)"
        }
    }

