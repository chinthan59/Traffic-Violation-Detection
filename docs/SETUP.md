# Setup Guide

Complete guide to set up and run the Traffic Violation Detection System.

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space for models and dependencies
- **GPU** (Optional but Recommended): 
  - NVIDIA GPU with CUDA support for faster inference
  - CUDA 11.8 or higher
  - cuDNN 8.6 or higher

### Software Requirements
- Python package manager: `pip` (usually comes with Python)
- Virtual environment tool: `venv` (included with Python 3.3+)

## Installation Steps

### 1. Clone or Navigate to Project Directory

```bash
cd /path/to/traffic_violation
```

### 2. Create Virtual Environment (Recommended)

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PARSeq (Text Recognition)

If you have a local `parseq` directory, install it:

```bash
pip install -e ./parseq/parseq
```

Alternatively, if PARSeq is available on PyPI:

```bash
pip install strhub
```

### 5. Verify Model Files

Ensure all model files are present in the `models/` directory:

```bash
ls models/
```

You should see:
- `car.pt` - Main car detection model (~49 MB)
- `car_yolov11.pt` - Seatbelt detection model (~6 MB)
- `bike.pt` - Bike violation detection model (~39 MB)

### 6. Verify CRNN Model

Check if the CRNN model exists:

```bash
ls src/fine_tuned_crnn.pth
```

This is an optional fallback for number plate recognition.

## Running the Application

### Development Mode

Start the FastAPI server with auto-reload:

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

### Production Mode

For production deployment, use a production ASGI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## GPU Setup (Optional)

### Install CUDA-enabled PyTorch

If you have an NVIDIA GPU, install CUDA-enabled PyTorch for faster inference:

1. **Check CUDA version:**
   ```bash
   nvidia-smi
   ```

2. **Install PyTorch with CUDA:**
   Visit [PyTorch website](https://pytorch.org/get-started/locally/) and install the appropriate version:
   
   Example for CUDA 11.8:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU availability:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` for `strhub` or other packages

**Solution**:
```bash
pip install -r requirements.txt
pip install -e ./parseq/parseq
```

#### 2. Model Not Found

**Problem**: `FileNotFoundError` for model files

**Solution**: Ensure model files are in the `models/` directory:
```bash
mkdir -p models
# Copy your model files to models/
```

#### 3. CUDA Out of Memory

**Problem**: GPU memory errors during inference

**Solution**: 
- Use CPU mode (models will automatically fall back)
- Process smaller images
- Reduce batch size in model configuration

#### 4. Port Already in Use

**Problem**: `Address already in use` error

**Solution**: Use a different port:
```bash
uvicorn main:app --port 8001
```

#### 5. Slow Inference on CPU

**Problem**: Very slow processing times

**Solution**:
- Use GPU if available
- Reduce image resolution before sending to API
- Consider using a more powerful CPU

## Environment Variables

You can configure the application using environment variables:

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
```

## Docker Setup (Optional)

If you prefer Docker, create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t traffic-violation .
docker run -p 8000:8000 traffic-violation
```

## Verification

### Test the API

1. **Check root endpoint:**
   ```bash
   curl http://localhost:8000/
   ```

2. **Test car detection:**
   ```bash
   curl -X POST "http://localhost:8000/analyze-car/" \
     -F "file=@test_car_image.jpg"
   ```

3. **Test bike detection:**
   ```bash
   curl -X POST "http://localhost:8000/analyze-bike/" \
     -F "file=@test_bike_image.jpg"
   ```

### Check Logs

Monitor the console output for any errors or warnings during startup and inference.

## Next Steps

- Read [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for API usage
- Check [USAGE.md](./USAGE.md) for code examples
- Review [ARCHITECTURE.md](./ARCHITECTURE.md) to understand the system design

