import os
import cv2
import glob
import shutil
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your Roboflow dataset root
DATASET_ROOT = r"C:\Users\ASUS\Desktop\Violation_Detection\bike_folders\Bike" 

# Output directory for the classification dataset
OUTPUT_DIR = "dataset_classifier"

# Class ID mapping (Based on your YOLO model)
CLASS_MAPPING = {
    2: 'violation', # mobile_usage
    3: 'violation', # pillion_not_wearing_helmet
    4: 'safe',      # pillion_wearing_helmet
    5: 'violation', # rider_not_wearing_helmet
    6: 'safe',      # rider_wearing_helmet
    7: 'violation'  # triple_riding
}

def process_subset(subset_name):
    """
    Process a subset (train/valid/test) and extract crops.
    """
    # Handle potential typo (trian vs train)
    target_dir = os.path.join(DATASET_ROOT, subset_name)
    if not os.path.exists(target_dir) and subset_name == 'train':
        if os.path.exists(os.path.join(DATASET_ROOT, 'trian')):
            target_dir = os.path.join(DATASET_ROOT, 'trian')
            print(f"Dataset note: Using 'trian' folder for '{subset_name}'")
    
    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Skipping {subset_name}: Folder {images_dir} or {labels_dir} not found.")
        return

    print(f"Processing {subset_name}...")
    
    # Target output folders (map 'valid' and 'test' to 'val' for standard structure if preferred, 
    # but keeping distinct names is fine too. PyTorch ImageFolder expects 'train' and 'val')
    # We will map 'valid' -> 'val' for compatibility with training script
    output_subset_name = 'val' if subset_name in ['valid', 'test'] else 'train'
    
    out_safe = os.path.join(OUTPUT_DIR, output_subset_name, "safe")
    out_viol = os.path.join(OUTPUT_DIR, output_subset_name, "violation")
    os.makedirs(out_safe, exist_ok=True)
    os.makedirs(out_viol, exist_ok=True)
    
    image_files = glob.glob(os.path.join(images_dir, "*.*"))
    print(f"Found {len(image_files)} images in {subset_name}")
    
    count_safe = 0
    count_viol = 0
    
    for idx, img_path in enumerate(image_files):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(image_files)} images...", end='\r')
            
        # Find corresponding label file
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            
            if cls_id not in CLASS_MAPPING:
                continue
                
            category = CLASS_MAPPING[cls_id]
            
            # Parse YOLO format (center_x, center_y, width, height) - normalized
            try:
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            
            # Convert to pixel coordinates
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Crop
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            # Save crop - use unique name to avoid overwrites
            # {subset}_{basename}_{index}_{class}.jpg
            save_name = f"{subset_name}_{basename}_{i}_{cls_id}.jpg"
            if category == 'safe':
                cv2.imwrite(os.path.join(out_safe, save_name), crop)
                count_safe += 1
            else:
                cv2.imwrite(os.path.join(out_viol, save_name), crop)
                count_viol += 1

    print(f"\nFinished {subset_name}: {count_safe} safe crops, {count_viol} violation crops.")

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root not found at {DATASET_ROOT}")
        print("Please check the path.")
        return

    # Process subsets
    process_subset("train")
    process_subset("valid")
    process_subset("test")
    
    print("\nDataset preparation complete!")
    print(f"Train/Val dataset created at: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Run this command to train:\npython src/train_verifier.py --train --data_dir {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
