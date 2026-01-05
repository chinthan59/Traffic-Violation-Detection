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
from src.detector2 import process_bike_image, process_bike_image_with_annotation

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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Traffic Violation Detection API",
        "endpoints": {
            "/analyze-car/": "POST - Analyze car images for seatbelt violations",
            "/analyze-bike/": "POST - Analyze bike images for helmet and other violations",
            "/analyze-bike-annotated/": "POST - Analyze bike images and return annotated image (uses both bike_old.pt and bike.pt models)"
        }
    }

