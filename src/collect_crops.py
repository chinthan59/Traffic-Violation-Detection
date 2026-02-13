import os
import cv2
import glob
import uuid
from ultralytics import YOLO

# Configuration
# Path to your images (can be a folder of images or a video)
SOURCE_PATH = "bike_folders/images"  # Change this to your source folder
OUTPUT_DIR = "collected_crops"
MODEL_PATH = "models/bike_yolov11_new.pt"

# Classes to crop (Violations you want to verify)
TARGET_CLASSES = [
    'pillion_not_wearing_helmet', 
    'rider_not_wearing_helmet',
    'triple_riding'
]

def collect_crops():
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get images
    if os.path.isdir(SOURCE_PATH):
        image_files = glob.glob(os.path.join(SOURCE_PATH, "*.*"))
    else:
        print(f"Source path {SOURCE_PATH} not found.")
        return

    print(f"Processing {len(image_files)} images...")
    
    count = 0
    for img_path in image_files:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            results = model.predict(img, conf=0.15, verbose=False)
            names = results[0].names
            boxes = results[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls.item())
                label = names[cls_id]
                
                if label in TARGET_CLASSES:
                    # Crop
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Add margin (optional)
                    h, w = img.shape[:2]
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(w, x2 + 10)
                    y2 = min(h, y2 + 10)
                    
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    # Save crop
                    filename = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(save_path, crop)
                    count += 1
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print(f"Done! Collected {count} crops in '{OUTPUT_DIR}'.")
    print("Now, manually sort these into 'dataset/train/violation' and 'dataset/train/safe'.")

if __name__ == "__main__":
    collect_crops()
