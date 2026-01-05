# Usage Examples

Practical examples for using the Traffic Violation Detection API.

## Table of Contents

- [Python Examples](#python-examples)
- [cURL Examples](#curl-examples)
- [JavaScript/Node.js Examples](#javascriptnodejs-examples)
- [Batch Processing](#batch-processing)
- [Error Handling](#error-handling)

## Python Examples

### Basic Car Detection

```python
import requests

# API endpoint
url = "http://localhost:8000/analyze-car/"

# Open and send image
with open("car_image.jpg", "rb") as f:
    files = {"file": ("car_image.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Image ID: {result['image_id']}")
        
        for detection in result['detections']:
            plate = detection.get('number_plate', {})
            print(f"Plate Number: {plate.get('plate_number', 'Not detected')}")
            
            for violation in detection.get('violations', []):
                print(f"  - {violation['type']}: {violation['confidence']:.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
```

### Basic Bike Detection

```python
import requests

url = "http://localhost:8000/analyze-bike/"

with open("bike_image.jpg", "rb") as f:
    files = {"file": ("bike_image.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        
        for detection in result['detections']:
            print(f"Plate: {detection.get('plate_number', 'Not detected')}")
            
            violations = detection.get('violations', [])
            if violations:
                print("Violations:")
                for v in violations:
                    print(f"  - {v['type']} (confidence: {v['confidence']:.2f})")
            else:
                print("No violations detected")
```

### Bike Detection with Annotated Image

```python
import requests

url = "http://localhost:8000/analyze-bike-annotated/"

with open("bike_image.jpg", "rb") as f:
    files = {"file": ("bike_image.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        # Save annotated image
        with open("annotated_bike.jpg", "wb") as out:
            out.write(response.content)
        print("Annotated image saved to annotated_bike.jpg")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
```

### Advanced: Processing Multiple Images

```python
import requests
import os
from pathlib import Path

def analyze_car_image(image_path):
    """Analyze a single car image."""
    url = "http://localhost:8000/analyze-car/"
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files)
        return response.json() if response.status_code == 200 else None

def process_directory(directory_path, vehicle_type="car"):
    """Process all images in a directory."""
    url = f"http://localhost:8000/analyze-{vehicle_type}/"
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    results = []
    image_dir = Path(directory_path)
    
    for image_file in image_dir.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"Processing: {image_file.name}")
            
            with open(image_file, "rb") as f:
                files = {"file": (image_file.name, f, "image/jpeg")}
                response = requests.post(url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    result['filename'] = image_file.name
                    results.append(result)
                else:
                    print(f"  Error: {response.status_code}")
    
    return results

# Usage
results = process_directory("./test_images", vehicle_type="car")
for result in results:
    print(f"\n{result['filename']}:")
    print(f"  Plate: {result['detections'][0].get('number_plate', {}).get('plate_number', 'N/A')}")
    print(f"  Violations: {len(result['detections'][0].get('violations', []))}")
```

### Save Results to File

```python
import requests
import json
from datetime import datetime

def analyze_and_save(image_path, output_file="results.json"):
    """Analyze image and save results to JSON file."""
    url = "http://localhost:8000/analyze-car/"
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            result['source_image'] = image_path
            result['analyzed_at'] = datetime.now().isoformat()
            
            # Save to file
            with open(output_file, 'w') as out:
                json.dump(result, out, indent=2)
            
            print(f"Results saved to {output_file}")
            return result
        else:
            print(f"Error: {response.status_code}")
            return None

# Usage
analyze_and_save("car_image.jpg", "car_results.json")
```

## cURL Examples

### Car Detection

```bash
# Basic request
curl -X POST "http://localhost:8000/analyze-car/" \
  -F "file=@car_image.jpg"

# Save response to file
curl -X POST "http://localhost:8000/analyze-car/" \
  -F "file=@car_image.jpg" \
  -o response.json

# Pretty print JSON response
curl -X POST "http://localhost:8000/analyze-car/" \
  -F "file=@car_image.jpg" | python -m json.tool
```

### Bike Detection

```bash
curl -X POST "http://localhost:8000/analyze-bike/" \
  -F "file=@bike_image.jpg" | python -m json.tool
```

### Bike Detection with Annotated Image

```bash
curl -X POST "http://localhost:8000/analyze-bike-annotated/" \
  -F "file=@bike_image.jpg" \
  --output annotated_bike.jpg
```

### Batch Processing with Shell Script

```bash
#!/bin/bash

# Process all images in a directory
for image in ./test_images/*.jpg; do
    echo "Processing: $image"
    curl -X POST "http://localhost:8000/analyze-car/" \
      -F "file=@$image" \
      -o "results/$(basename $image .jpg).json"
done
```

## JavaScript/Node.js Examples

### Using Fetch API (Browser)

```javascript
async function analyzeCar(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
        const response = await fetch('http://localhost:8000/analyze-car/', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('Detections:', result.detections);
            return result;
        } else {
            console.error('Error:', response.status);
            return null;
        }
    } catch (error) {
        console.error('Request failed:', error);
        return null;
    }
}

// Usage with file input
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        const result = await analyzeCar(file);
        console.log(result);
    }
});
```

### Using Node.js with form-data

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function analyzeCar(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    
    try {
        const response = await axios.post(
            'http://localhost:8000/analyze-car/',
            form,
            {
                headers: form.getHeaders()
            }
        );
        
        return response.data;
    } catch (error) {
        console.error('Error:', error.message);
        return null;
    }
}

// Usage
analyzeCar('./car_image.jpg').then(result => {
    console.log(JSON.stringify(result, null, 2));
});
```

## Batch Processing

### Python Batch Processor Class

```python
import requests
import json
from pathlib import Path
from typing import List, Dict

class TrafficViolationAnalyzer:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.car_endpoint = f"{base_url}/analyze-car/"
        self.bike_endpoint = f"{base_url}/analyze-bike/"
    
    def analyze_car(self, image_path: str) -> Dict:
        """Analyze a single car image."""
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f)}
            response = requests.post(self.car_endpoint, files=files)
            return response.json() if response.status_code == 200 else None
    
    def analyze_bike(self, image_path: str) -> Dict:
        """Analyze a single bike image."""
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f)}
            response = requests.post(self.bike_endpoint, files=files)
            return response.json() if response.status_code == 200 else None
    
    def batch_analyze(self, image_paths: List[str], vehicle_type: str = "car") -> List[Dict]:
        """Analyze multiple images."""
        results = []
        endpoint = self.car_endpoint if vehicle_type == "car" else self.bike_endpoint
        
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f)}
                response = requests.post(endpoint, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    result['source_file'] = image_path
                    results.append(result)
                else:
                    print(f"  Error: {response.status_code}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save batch results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {output_file}")

# Usage
analyzer = TrafficViolationAnalyzer()

# Single image
result = analyzer.analyze_car("car1.jpg")
print(result)

# Batch processing
images = ["car1.jpg", "car2.jpg", "car3.jpg"]
results = analyzer.batch_analyze(images, vehicle_type="car")
analyzer.save_results(results, "batch_results.json")
```

## Error Handling

### Comprehensive Error Handling

```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

def safe_analyze_car(image_path, timeout=30):
    """Analyze car image with comprehensive error handling."""
    url = "http://localhost:8000/analyze-car/"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f)}
            response = requests.post(url, files=files, timeout=timeout)
            
            # Check HTTP status
            response.raise_for_status()
            
            # Parse JSON
            result = response.json()
            
            # Validate response structure
            if result.get('status') != 'success':
                print(f"Warning: Unexpected status: {result.get('status')}")
            
            return result
            
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    
    except ConnectionError:
        print("Error: Could not connect to API server. Is it running?")
        return None
    
    except Timeout:
        print(f"Error: Request timed out after {timeout} seconds")
        return None
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None
    
    except ValueError as e:
        print(f"JSON Parse Error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

# Usage
result = safe_analyze_car("car_image.jpg")
if result:
    print("Analysis successful!")
    print(result)
```

## Integration Examples

### Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
VIOLATION_API = "http://localhost:8000"

@app.route('/check-violation', methods=['POST'])
def check_violation():
    """Proxy endpoint that adds additional processing."""
    vehicle_type = request.form.get('type', 'car')
    file = request.files.get('file')
    
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    # Forward to violation detection API
    endpoint = f"{VIOLATION_API}/analyze-{vehicle_type}/"
    files = {'file': (file.filename, file.stream, file.content_type)}
    
    response = requests.post(endpoint, files=files)
    
    if response.status_code == 200:
        result = response.json()
        # Add custom processing here
        result['processed_by'] = 'custom_service'
        return jsonify(result)
    else:
        return jsonify({'error': 'Detection failed'}), 500

if __name__ == '__main__':
    app.run(port=5000)
```

### Django Integration

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json

@csrf_exempt
def analyze_vehicle(request):
    if request.method == 'POST':
        vehicle_type = request.POST.get('type', 'car')
        file = request.FILES.get('file')
        
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        # Forward to API
        url = f"http://localhost:8000/analyze-{vehicle_type}/"
        files = {'file': (file.name, file.read(), file.content_type)}
        
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Detection failed'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
```

## Tips and Best Practices

1. **Image Quality**: Use high-resolution images for better accuracy
2. **File Size**: Compress very large images before sending
3. **Error Handling**: Always implement proper error handling
4. **Timeouts**: Set appropriate timeout values for requests
5. **Batch Processing**: Process images sequentially to avoid overwhelming the server
6. **Caching**: Cache results if processing the same images multiple times
7. **Logging**: Log all API interactions for debugging

