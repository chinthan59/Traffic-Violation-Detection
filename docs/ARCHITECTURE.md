# System Architecture

Overview of the Traffic Violation Detection System architecture and design.

## High-Level Architecture

```
┌─────────────────┐
│   Client/User   │
└────────┬────────┘
         │ HTTP Request (Image)
         ▼
┌─────────────────────────────────┐
│      FastAPI Application         │
│         (main.py)                │
└────────┬────────────────────────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Car Detector │  │ Bike Detector │  │   OCR       │
│ (detector.py)│  │(detector2.py) │  │ (PARSeq/    │
└──────┬───────┘  └──────┬────────┘  │  CRNN)      │
       │                 │            └─────────────┘
       │                 │
       ├─────────┬───────┤
       ▼         ▼       ▼
┌─────────────────────────────────┐
│      YOLO Models                 │
│  - car.pt                        │
│  - car_yolov11.pt                │
│  - bike.pt                       │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│      Response (JSON)            │
│  - Detections                   │
│  - Violations                   │
│  - Plate Numbers                 │
└─────────────────────────────────┘
```

## Component Details

### 1. API Layer (`main.py`)

**FastAPI Application** that provides REST endpoints:
- Handles HTTP requests
- Manages file uploads
- Routes requests to appropriate detectors
- Returns JSON responses

**Key Features:**
- Async request handling
- Automatic request validation
- Interactive API documentation
- Error handling

### 2. Car Detection Module (`src/detector.py`)

**Dual Model Architecture** for car violation detection:

#### Models Used:
1. **car.pt** (YOLOv8)
   - Main detection model
   - Detects: cars, number plates, seatbelt violations
   - Classes: 0=Cars, 1=Driver_not_wearing_seatbelt, 2=Driver_wearing_seatbelt, 3=Number_plate, 4=passenger_not_wearing_seatbelt, 5=seat_belt, 6=vehicle_with_offense

2. **car_yolov11.pt** (YOLOv11 Medium)
   - Specialized seatbelt detection
   - Classes: 0=Driver_not_wearing_seatbelt, 1=Driver_wearing_seatbelt, 2=Number_plate, 3=car, 4=passenger-wearing-seat_belt, 5=passenger_not_wearing_seatbelt

**Detection Pipeline:**
```
1. Image Preprocessing
   ├─ Load image
   ├─ Detect lighting (day/night)
   └─ Detect view type (aerial/side)

2. Dual Model Inference
   ├─ Run car.pt
   └─ Run car_yolov11.pt

3. Violation Filtering
   ├─ Filter false positives using "wearing" detections
   ├─ Deduplicate overlapping violations
   └─ Validate confidence thresholds

4. Plate Detection & OCR
   ├─ Detect number plates
   ├─ Crop plate regions
   ├─ PARSeq OCR (primary)
   └─ CRNN OCR (fallback)

5. Association
   ├─ Match violations to plates
   └─ Handle unmatched violations

6. Response Formatting
   └─ Structure JSON response
```

**Adaptive Features:**
- **Day/Night Detection**: Adjusts confidence thresholds based on lighting
- **View Type Detection**: Adapts for aerial vs side views
- **False Positive Reduction**: Uses "wearing seatbelt" detections to filter contradictions
- **Deduplication**: Removes overlapping detections using IoU and containment checks

### 3. Bike Detection Module (`src/detector2.py`)

**Single Model Architecture** with conflict resolution and vehicle grouping:

#### Model Used:
- **bike.pt** (YOLOv11)
   - Detects bike violations and compliance
   - Classes: 0=Number_plate, 1=mobile_usage, 2=pillion_not_wearing_helmet, 3=pillion_wearing_helmet, 4=rider_not_wearing_helmet, 5=rider_wearing_helmet, 6=triple_riding, 7=vehicle

**Detection Pipeline:**
```
1. Image Loading & Preprocessing
   ├─ Load image
   ├─ Detect lighting (day/night)
   └─ Set adaptive confidence thresholds

2. YOLO Inference
   └─ Run bike.pt model

3. Collect Detections
   ├─ Separate violations (not wearing)
   ├─ Separate compliance (wearing)
   ├─ Separate vehicles
   └─ Separate plates

4. Conflict Resolution (NEW!)
   ├─ Check for overlapping "wearing" vs "not wearing"
   ├─ Compare confidence scores
   └─ Keep only higher confidence detection
   └─ Prevents contradictory detections

5. Deduplication
   ├─ Remove overlapping violations
   └─ Use IoU and containment checks

6. Vehicle Grouping (NEW!)
   ├─ Group detections by vehicle
   ├─ Associate violations to vehicles
   └─ Handle unmatched detections

7. Plate Detection & OCR
   ├─ Detect number plates (class 0)
   ├─ Crop plate regions
   ├─ PARSeq OCR (primary)
   └─ CRNN OCR (fallback)

8. Association
   ├─ Match violations to plates via vehicles
   └─ Handle spatial relationships

9. Response Formatting
   └─ Structure JSON response with vehicle groups
```

### 4. OCR Module

**Text Recognition** for number plates:

#### Primary: PARSeq
- State-of-the-art scene text recognition
- Handles 2-line Indian number plates
- Removes "IND" strip automatically
- Validates against Indian plate format regex

#### Fallback: CRNN
- Custom fine-tuned CRNN model
- Used when PARSeq fails or unavailable
- Character set: `-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`

**OCR Pipeline:**
```
1. Plate Cropping
   └─ Extract plate region from image

2. Preprocessing
   ├─ Remove IND strip (if present)
   ├─ Split 2-line plates (if needed)
   └─ Normalize image

3. Text Recognition
   ├─ Try PARSeq (primary)
   └─ Try CRNN (fallback)

4. Post-processing
   ├─ Clean OCR output
   ├─ Remove junk characters
   ├─ Validate state codes
   └─ Format plate number
```

## Data Flow

### Request Flow
```
Client Request
    │
    ├─> FastAPI receives image
    │
    ├─> Save to temporary file
    │
    ├─> Route to detector (car/bike)
    │
    ├─> Load and preprocess image
    │
    ├─> Run YOLO models
    │
    ├─> Post-process detections
    │
    ├─> OCR for number plates
    │
    ├─> Associate violations to plates
    │
    ├─> Format response
    │
    └─> Return JSON + cleanup temp file
```

### Response Structure
```json
{
  "status": "success",
  "timestamp": "ISO timestamp",
  "image_id": "unique_id",
  "detections": [
    {
      "number_plate": {
        "plate_number": "string",
        "plate_bbox": [x1, y1, x2, y2]
      },
      "violations": [
        {
          "type": "violation_type",
          "confidence": 0.0-1.0,
          "bbox": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

## Key Algorithms

### 1. IoU (Intersection over Union)
Used for deduplication and overlap detection:
```python
IoU = Intersection Area / (Area1 + Area2 - Intersection Area)
```

### 2. Distance-Based Association
For matching violations to plates:
```python
distance = sqrt((v_center_x - p_center_x)² + (v_center_y - p_center_y)²)
```

### 3. Confidence Thresholding
Adaptive thresholds based on conditions:
- Day images: 0.55-0.60
- Night images: 0.50
- Aerial views: 0.55

### 4. Conflict Resolution (Bike Detection)
Resolves contradictory detections for the same person:
- Checks for overlapping "wearing" and "not wearing" detections
- Uses IoU threshold of 0.3 to determine if same person
- Compares confidence scores
- Keeps only the detection with higher confidence
- Prevents both "wearing" and "not wearing" from appearing simultaneously

**Conflicting Pairs:**
- `pillion_not_wearing_helmet` vs `pillion_wearing_helmet`
- `rider_not_wearing_helmet` vs `rider_wearing_helmet`

### 5. False Positive Filtering (Car Detection)
Uses "wearing" detections to filter contradictions:
- If violation overlaps with "wearing" detection
- And "wearing" confidence >= (violation confidence - 0.15)
- Then filter as false positive

## Performance Optimizations

1. **Model Loading**: Models loaded once at startup
2. **GPU Acceleration**: Automatic CUDA detection and usage
3. **Adaptive Processing**: Adjusts based on image conditions
4. **Deduplication**: Reduces redundant detections
5. **Early Exit**: Returns early if no violations detected

## Error Handling

- **Model Loading Errors**: Graceful fallback
- **OCR Failures**: Multiple OCR methods
- **Image Processing Errors**: Try-catch blocks with error messages
- **File I/O Errors**: Temporary file cleanup in finally blocks

## Scalability Considerations

- **Stateless API**: Can be horizontally scaled
- **Model Caching**: Models loaded in memory
- **Async Processing**: FastAPI async support
- **Batch Processing**: Can be extended for batch inference

## Security Considerations

- **File Validation**: Filename sanitization
- **Temporary Files**: Automatic cleanup
- **Input Validation**: FastAPI automatic validation
- **Error Messages**: Don't expose sensitive information

