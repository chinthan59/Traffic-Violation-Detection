# Models Documentation

Detailed information about the machine learning models used in the Traffic Violation Detection System.

## Model Overview

The system uses YOLO (You Only Look Once) object detection models for detecting vehicles, violations, and number plates.

## Car Detection Models

### 1. car.pt (Main Car Model)

**Type**: YOLOv8  
**Size**: ~49 MB  
**Purpose**: Primary detection model for cars and violations

**Classes Detected:**
- `0`: Cars
- `1`: Driver_not_wearing_seatbelt
- `2`: Driver_wearing_seatbelt
- `3`: Number_plate
- `4`: passenger_not_wearing_seatbelt
- `5`: seat_belt
- `6`: vehicle_with_offense (generic)

**Usage**: Primary model for detecting cars, number plates, and seatbelt violations.

**Location**: `models/car.pt`

---

### 2. car_yolov11.pt (Seatbelt Detection Model)

**Type**: YOLOv11 Medium  
**Size**: ~6 MB  
**Purpose**: Specialized seatbelt detection with higher precision

**Classes Detected:**
- `0`: Driver_not_wearing_seatbelt
- `1`: Driver_wearing_seatbelt
- `2`: Number_plate
- `3`: car
- `4`: passenger-wearing-seat_belt
- `5`: passenger_not_wearing_seatbelt

**Usage**: Secondary model for improved seatbelt detection accuracy. Used in conjunction with `car.pt` for dual-model detection.

**Location**: `models/car_yolov11.pt`

**Key Features:**
- Higher precision for seatbelt detection
- Better handling of edge cases
- Validates against "wearing seatbelt" detections to reduce false positives

---

## Bike Detection Model

### 3. bike.pt (Bike Violation Model)

**Type**: YOLOv11  
**Size**: ~39 MB  
**Purpose**: Detection of bike violations and number plates

**Classes Detected:**
- `0`: Number_plate
- `1`: mobile_usage
- `2`: pillion_not_wearing_helmet
- `3`: pillion_wearing_helmet
- `4`: rider_not_wearing_helmet
- `5`: rider_wearing_helmet
- `6`: triple_riding
- `7`: vehicle

**Usage**: Single model for all bike-related violation detection.

**Location**: `models/bike.pt`

**Key Features:**
- **Conflict Resolution**: Automatically resolves conflicts between "wearing" and "not wearing" detections
- **Vehicle Grouping**: Groups detections by vehicle for better organization
- **Adaptive Thresholds**: Adjusts based on day/night conditions

**Violation Types:**
1. **Helmet Violations**:
   - Rider not wearing helmet (class 4)
   - Pillion rider not wearing helmet (class 2)
   - Both rider and pillion not wearing helmet (legacy)

2. **Compliance Types** (not violations):
   - Rider wearing helmet (class 5)
   - Pillion wearing helmet (class 3)

3. **Other Violations**:
   - Mobile phone usage while riding (class 1)
   - Triple riding (3+ people on bike) (class 6)

**Conflict Resolution Logic:**
- If both "rider_not_wearing_helmet" and "rider_wearing_helmet" are detected for the same person (IoU > 0.3), only the detection with higher confidence is kept
- Same logic applies for pillion rider detections

---

## OCR Models

### PARSeq (Primary OCR)

**Type**: Scene Text Recognition  
**Purpose**: Number plate text recognition

**Features:**
- Handles 2-line Indian number plates
- Automatic "IND" strip removal
- Validates against Indian plate format
- High accuracy for clear images

**Installation**: 
```bash
pip install -e ./parseq/parseq
# or
pip install strhub
```

**Usage**: Primary method for OCR in both car and bike detection.

---

### CRNN (Fallback OCR)

**Type**: Convolutional Recurrent Neural Network  
**Purpose**: Fallback OCR when PARSeq fails

**Model File**: `src/fine_tuned_crnn.pth`

**Character Set**: `-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`

**Features:**
- Fine-tuned for Indian number plates
- Used as backup when PARSeq unavailable
- Handles single-line plates well

**Architecture**:
- Feature Extraction: VGG
- Sequence Modeling: BiLSTM
- Prediction: CTC

---

## Model Configuration

### Detection Parameters

#### Car Detection
- **Confidence Thresholds**:
  - Day images (aerial): 0.55
  - Day images (side): 0.60
  - Night images: 0.50

- **Image Sizes**:
  - Day: 960px
  - Night: 640px

- **Augmentation**: Disabled (to reduce false positives)

#### Bike Detection
- **Confidence Threshold**: 0.25 (default)
- **Image Size**: Model default (640px)

### Post-Processing

#### Deduplication
- **IoU Threshold**: 0.5
- **Containment Threshold**: 0.8
- Removes overlapping detections

#### False Positive Filtering
- Checks for "wearing" detections
- Filters violations that overlap with "wearing" detections
- Confidence difference threshold: 0.15

---

## Model Performance

### Accuracy Considerations

1. **Lighting Conditions**:
   - Day images: Higher accuracy
   - Night images: Slightly lower, but adaptive thresholds help

2. **View Angles**:
   - Side/front views: Best accuracy
   - Aerial views: Good accuracy with adjusted parameters

3. **Image Quality**:
   - High resolution: Better results
   - Low resolution: May miss small violations

### Limitations

1. **Occlusion**: May miss violations if partially occluded
2. **Angle**: Extreme angles may reduce accuracy
3. **Lighting**: Very dark or overexposed images may affect results
4. **Distance**: Very far vehicles may have lower detection rates

---

## Model Updates

To update or replace models:

1. **Place new model file** in `models/` directory
2. **Update path** in detector code:
   - `src/detector.py` for car models
   - `src/detector2.py` for bike model
3. **Restart the API server**

### Model Naming Convention

- `car.pt`: Main car detection model
- `car_yolov11.pt`: Seatbelt detection model
- `bike.pt`: Bike violation detection model

---

## Hardware Requirements

### Minimum (CPU Only)
- **RAM**: 8GB
- **CPU**: Multi-core processor
- **Storage**: 5GB for models and dependencies

### Recommended (GPU)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.6 or higher
- **RAM**: 16GB

### Performance

- **CPU Inference**: ~2-5 seconds per image
- **GPU Inference**: ~0.5-1 second per image

---

## Model Training

If you need to retrain models:

1. **Data Preparation**: Annotate images with bounding boxes
2. **YOLO Training**: Use Ultralytics YOLO training pipeline
3. **Validation**: Test on validation set
4. **Export**: Save best model as `.pt` file
5. **Deploy**: Replace model file and restart

For more details on training, refer to [Ultralytics YOLO documentation](https://docs.ultralytics.com/).

---

## Model Versioning

Current model versions:
- `car.pt`: Version 1.0
- `car_yolov11.pt`: Version 1.0
- `bike.pt`: Version 1.0

When updating models, consider versioning:
- `car_v2.pt`
- `bike_v2.pt`

Update code references accordingly.

---

## Troubleshooting

### Model Loading Errors

**Problem**: `FileNotFoundError` for model file

**Solution**: 
- Verify model file exists in `models/` directory
- Check file permissions
- Ensure correct path in code

### Low Detection Accuracy

**Solutions**:
- Adjust confidence thresholds
- Improve image quality
- Check lighting conditions
- Verify model is appropriate for use case

### Memory Issues

**Solutions**:
- Use smaller image sizes
- Process images in batches
- Use GPU if available
- Reduce model complexity

