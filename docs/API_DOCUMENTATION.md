# API Documentation

Complete API reference for the Traffic Violation Detection System.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Traffic Violation Detection API",
  "endpoints": {
    "/analyze-car/": "POST - Analyze car images for seatbelt violations",
    "/analyze-bike/": "POST - Analyze bike images for helmet and other violations",
    "/analyze-bike-annotated/": "POST - Analyze bike images and return annotated image"
  }
}
```

---

### 2. Car Detection

**POST** `/analyze-car/`

Analyzes car images for seatbelt violations and number plate recognition.

**Request:**
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (required): Image file (JPEG, PNG, etc.)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/analyze-car/" \
  -F "file=@path/to/car_image.jpg"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/analyze-car/"
with open("car_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())
```

**Response Format:**
```json
{
  "status": "success",
  "timestamp": "2024-01-02T12:00:00.000000Z",
  "image_id": "IMG_20240102_120000_abc123",
  "detections": [
    {
      "number_plate": {
        "plate_number": "KA03MR4902",
        "plate_bbox": [100, 200, 300, 250]
      },
      "violations": [
        {
          "type": "Driver_not_wearing_seatbelt",
          "confidence": 0.85,
          "bbox": [150, 100, 250, 200]
        }
      ]
    }
  ]
}
```

**Response Fields:**
- `status`: Always "success" for successful requests
- `timestamp`: UTC timestamp in ISO format
- `image_id`: Unique identifier for the processed image
- `detections`: Array of detection objects
  - `number_plate`: Plate information
    - `plate_number`: Recognized plate text (empty if not detected)
    - `plate_bbox`: Bounding box coordinates `[x1, y1, x2, y2]`
  - `violations`: Array of violation objects
    - `type`: Violation type (e.g., "Driver_not_wearing_seatbelt")
    - `confidence`: Detection confidence score (0.0 to 1.0)
    - `bbox`: Bounding box coordinates `[x1, y1, x2, y2]`

**Possible Violation Types:**
- `Driver_not_wearing_seatbelt`
- `passenger_not_wearing_seatbelt`

**Error Response:**
```json
{
  "detail": "Error message description"
}
```

---

### 3. Bike Detection

**POST** `/analyze-bike/`

Analyzes bike images for helmet violations, mobile usage, and other bike-related offenses.

**Request:**
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (required): Image file (JPEG, PNG, etc.)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/analyze-bike/" \
  -F "file=@path/to/bike_image.jpg"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/analyze-bike/"
with open("bike_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())
```

**Response Format:**
```json
{
  "status": "success",
  "timestamp": "2024-01-02T12:00:00.000000Z",
  "image_id": "BIKE_20240102_120000_abc123",
  "detections": [
    {
      "plate_number": "KA03MR4902",
      "plate_bbox": [100, 200, 300, 250],
      "violations": [
        {
          "type": "rider_not_wearing_helmet",
          "confidence": 0.92,
          "bbox": [150, 100, 250, 200]
        },
        {
          "type": "mobile_usage",
          "confidence": 0.78,
          "bbox": [160, 110, 200, 150]
        }
      ]
    }
  ]
}
```

**Response Fields:**
- `status`: Always "success" for successful requests
- `timestamp`: UTC timestamp in ISO format
- `image_id`: Unique identifier prefixed with "BIKE_"
- `detections`: Array of detection objects
  - `plate_number`: Recognized plate text (empty if not detected)
  - `plate_bbox`: Bounding box coordinates `[x1, y1, x2, y2]`
  - `violations`: Array of violation objects
    - `type`: Violation type
    - `confidence`: Detection confidence score (0.0 to 1.0)
    - `bbox`: Bounding box coordinates `[x1, y1, x2, y2]`

**Possible Violation Types:**
- `rider_not_wearing_helmet`
- `pillion_not_wearing_helmet`
- `pillion_rider_not_wearing_helmet` (legacy)
- `rider_and_pillion_not_wearing_helmet`
- `mobile_usage`
- `triple_riding`
- `vehicle_with_offence` (legacy)

**Note**: The system automatically resolves conflicts between "wearing" and "not wearing" detections. If both are detected for the same person, only the detection with higher confidence is kept.

**Error Response:**
```json
{
  "detail": "Error message description"
}
```

---

## Response Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request (e.g., missing file)
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server error during processing

## Notes

1. **Image Formats**: Supported formats include JPEG, PNG, and other common image formats
2. **File Size**: Large images may take longer to process
3. **Confidence Scores**: Higher confidence scores indicate more reliable detections
4. **Bounding Boxes**: Coordinates are in pixel values `[x1, y1, x2, y2]` where (x1, y1) is top-left and (x2, y2) is bottom-right
5. **Empty Detections**: If no violations or plates are detected, the `detections` array may be empty or contain entries with empty `plate_number` fields

## Interactive API Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to test the API directly from your browser.

