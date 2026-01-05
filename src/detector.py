# detector_carbest.py
# Dual model detector using car.pt (main) + car_yolov11.pt (seatbelt - YOLOv11 medium)
import os
import sys
import uuid
import re
from datetime import datetime

# Add parseq directory to Python path for strhub import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parseq_path = os.path.join(project_root, 'parseq')
if parseq_path not in sys.path:
    sys.path.insert(0, parseq_path)

import cv2
import numpy as np
from PIL import Image
import string
import torch
from argparse import Namespace

from ultralytics import YOLO
from src.model import Model
from strhub.data.module import SceneTextDataModule


# -------------------------
# Configs and model paths
# -------------------------
CHARACTER_SET = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')  # Indian plate format

# Valid Indian state/UT codes (first 2 characters of number plate)
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
    'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
    'TN', 'TS', 'TR', 'UK', 'UP', 'WB'
}

CRNN_MODEL_PATH = "src/fine_tuned_crnn.pth"
YOLO_MAIN_MODEL_PATH = "models/car.pt"              # Main model (YOLOv8)
YOLO_SEATBELT_MODEL_PATH = "models/car_yolov11.pt"   # Seatbelt-focused model (YOLOv11 medium)

CROPPED_PLATES_DIR = "folders/cropped_number_plates"
OFFENSES_DIR = "folders/vehicles_with_offense"
NO_OFFENSES_DIR = "folders/images_not_with_offense"

os.makedirs(CROPPED_PLATES_DIR, exist_ok=True)
os.makedirs(OFFENSES_DIR, exist_ok=True)
os.makedirs(NO_OFFENSES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# CRNN (optional fallback)
# -------------------------
class ANPRPredictor:
    def __init__(self, model_path, character_set):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.character = character_set

        opt = Namespace(
            input_channel=1,
            output_channel=512,
            hidden_size=256,
            num_class=len(self.character),
            FeatureExtraction='VGG',
            SequenceModeling='BiLSTM',
            Prediction='CTC',
            Transformation=None,
            PAD=False,
            num_lstm=2,
            num_fiducial=20,
            imgH=32,
            imgW=100,
            batch_max_length=50,
            saved_model=''
        )

        self.model = Model(opt).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.eval()
        except Exception as e:
            print(f"❌ Failed to load CRNN weights: {e}")
            self.model = None

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 100)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _decode_prediction(self, preds):
        _, pred_indices = preds.max(2)
        pred_indices = pred_indices.squeeze(0)
        last_char = None
        chars = []
        for idx in pred_indices:
            idx = idx.item()
            if idx != 0 and idx != last_char:
                chars.append(self.character[idx])
            last_char = idx
        return ''.join(chars)

    def predict(self, img_input):
        if self.model is None:
            return ""
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                return ""
        elif isinstance(img_input, np.ndarray):
            image = img_input
        else:
            raise TypeError("Invalid input")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor, text='', is_train=False)
        return self._decode_prediction(output)


try:
    crnn_predictor = ANPRPredictor(CRNN_MODEL_PATH, CHARACTER_SET)
except Exception as e:
    print(f"⚠️ CRNN not loaded: {e}")
    crnn_predictor = None

# Alias for backward compatibility (used by detector2.py)
predictor = crnn_predictor


# -------------------------
# Load PARSeq model (for plate recognition)
# -------------------------
def initialize_parseq(device):
    try:
        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval().to(device)
        transform = SceneTextDataModule.get_transform(model.hparams.img_size)
        print("✅ PARSeq loaded")
        return model, transform
    except Exception as e:
        print(f"❌ PARSeq load failed: {e}")
        return None, None


model_parseq, transform_parseq = initialize_parseq(device)


# -------------------------
# Helper functions for 2-line plate detection
# -------------------------
def remove_ind_strip(img):
    """
    Remove the blue 'IND' strip from Indian number plates.
    The strip is typically on the left 18% of the plate.
    """
    h, w = img.shape[:2]
    return img[:, int(0.18 * w):]


def split_two_lines(img):
    """
    Split a plate image into top and bottom halves for 2-line plates.
    Returns: (top_half, bottom_half)
    """
    h = img.shape[0]
    return img[:h//2], img[h//2:]


def clean_plate_number(plate_text):
    """
    Clean OCR output by removing junk characters before valid Indian state code.
    
    Strategy:
    1. Find ALL occurrences of valid 2-letter Indian state codes
    2. For each candidate, check if the remaining text matches plate format
    3. Choose the candidate that produces a valid plate format
    4. If none match format, return the first valid state code found
    5. If no state codes found, return original text
    
    Example:
        Input:  "SKA03MR4902"  (S is junk, SK is valid state but wrong)
        Output: "KA03MR4902"   (KA is correct state code, matches format)
    
    Args:
        plate_text: Raw OCR text (uppercase, alphanumeric only)
    
    Returns:
        Cleaned plate text starting from best valid state code
    """
    if not plate_text or len(plate_text) < 2:
        return plate_text
    
    # Find all valid state code positions
    candidates = []
    for i in range(len(plate_text) - 1):
        two_chars = plate_text[i:i+2]
        
        # Check if both characters are letters
        if two_chars.isalpha() and two_chars.isupper():
            # Check if it's a valid Indian state code
            if two_chars in INDIAN_STATE_CODES:
                trimmed = plate_text[i:]
                candidates.append((i, two_chars, trimmed))
    
    # If no valid state codes found, return original
    if not candidates:
        return plate_text
    
    # Priority 1: Find candidate that matches full plate regex
    for pos, state_code, trimmed in candidates:
        if PLATE_REGEX.match(trimmed):
            return trimmed
    
    # Priority 2: Find candidate that at least starts with correct pattern
    # (2 letters + 2 digits)
    for pos, state_code, trimmed in candidates:
        if len(trimmed) >= 4:
            # Check if format is: 2 letters + 2 digits + ...
            if (trimmed[0:2].isalpha() and 
                trimmed[2:4].isdigit()):
                return trimmed
    
    # Priority 3: Return first valid state code found
    return candidates[0][2]



def ocr_line(img):
    """
    Perform OCR on a single line of text using PARSeq.
    Returns cleaned uppercase text without spaces.
    """
    if model_parseq is None or transform_parseq is None:
        return ""
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = transform_parseq(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model_parseq(tensor)
        preds = logits.softmax(-1)
        labels, _ = model_parseq.tokenizer.decode(preds)
        if not labels:
            return ""
        return labels[0].replace(" ", "").upper()
    except Exception as e:
        return ""


def predict_plate_with_parseq(image_input):
    """
    Enhanced plate recognition with 2-line plate support.
    
    Strategy:
    1. Try reading full image first
    2. Try split approach (remove IND strip, split into top/bottom, concatenate)
    3. Clean both results (remove non-alphanumeric)
    4. Use regex validation to choose best result
    5. Fallback to result closer to standard length (10 chars)
    
    This handles both single-line and 2-line plates automatically.
    """
    if model_parseq is None or transform_parseq is None:
        return ""
    
    try:
        # Load image
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return ""
        else:
            img = image_input
        
        # Helper to clean OCR output
        clean = lambda s: re.sub(r'[^A-Z0-9]', '', s)
        
        # 1. Try full image OCR
        full_raw = ocr_line(img)
        full_text = clean(full_raw)
        full_text = clean_plate_number(full_text)  # Trim junk before state code
        
        # 2. Try split/processed version (remove IND strip, split lines)
        processed = remove_ind_strip(img)
        top, bottom = split_two_lines(processed)
        
        t1 = clean(ocr_line(top))
        t2 = clean(ocr_line(bottom))
        split_text = t1 + t2
        split_text = clean_plate_number(split_text)  # Trim junk before state code
        
        # 3. Choose best result using validation
        
        # Priority 1: Indian Regex Match
        full_match = bool(PLATE_REGEX.match(full_text))
        split_match = bool(PLATE_REGEX.match(split_text))
        
        if split_match and not full_match:
            return split_text
        if full_match and not split_match:
            return full_text
        if full_match and split_match:
            # Both valid - prefer one closer to standard length (10 chars)
            if abs(len(split_text) - 10) < abs(len(full_text) - 10):
                return split_text
            return full_text
        
        # Priority 2: Fallback (neither matches regex)
        # Prefer the one closer to expected length (10 chars)
        if abs(len(split_text) - 10) < abs(len(full_text) - 10):
            return split_text
        return full_text
        
    except Exception as e:
        print(f"PARSeq error: {e}")
        return ""


# -------------------------
# Load YOLO models
# -------------------------
try:
    yolo_main_model = YOLO(YOLO_MAIN_MODEL_PATH)
    yolo_seatbelt_model = YOLO(YOLO_SEATBELT_MODEL_PATH)
    print(f"✅ Main model loaded: {YOLO_MAIN_MODEL_PATH}")
    print(f"✅ Seatbelt model loaded: {YOLO_SEATBELT_MODEL_PATH}")
except Exception as e:
    print(f"❌ YOLO loading error: {e}")
    yolo_main_model = None
    yolo_seatbelt_model = None


# -------------------------
# Class mappings
# -------------------------
# Main model (car.pt)
# 0: Cars
# 1: Driver_not_wearing_seatbelt
# 2: Driver_wearing_seatbelt
# 3: Number_plate
# 4: passenger_not_wearing_seatbelt
# 5: seat_belt
# 6: vehicle with offense (generic)
MAIN_PLATE_CLASS = 3
MAIN_VIOLATION_CLASSES = [1, 4]  # Only no-seatbelt violations
MAIN_WEARING_CLASSES = [2]  # Driver wearing seatbelt (class 2)
MAIN_IGNORE_CLASSES = [6]  # Generic offense class

# Seatbelt model (car_yolov11.pt - YOLOv11 medium) - UPDATED CLASSES WITH NO_PASSENGER
# 0: Driver_not_wearing_seatbelt
# 1: Driver_wearing_seatbelt
# 2: Number_plate
# 3: car
# 4: no_passenger
# 5: passenger_not_wearing_seatbelt
# 6: passenger_wearing_seat_belt
SEATBELT_PLATE_CLASS = 2
SEATBELT_VIOLATION_CLASSES = [0, 5]  # Only no-seatbelt violations: Driver & passenger
SEATBELT_WEARING_CLASSES = [1, 6]  # Wearing seatbelt classes: Driver & passenger
SEATBELT_IGNORE_CLASSES = [3, 4]  # 3=car, 4=no_passenger (prevents false positives)


# -------------------------
# IOU helper
# -------------------------
def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


def deduplicate_violations(violations, iou_threshold=0.5, containment_threshold=0.8):
    """
    Remove duplicate violations using Non-Maximum Suppression with containment check.
    Prevents the same person from being reported multiple times for the same violation.
    
    Handles two cases:
    1. Overlapping boxes (IoU > threshold)
    2. Contained boxes (smaller box inside larger box)
    
    Args:
        violations: List of violation dicts with 'type', 'confidence', 'bbox'
        iou_threshold: IoU threshold for considering violations as duplicates (default: 0.5)
        containment_threshold: Containment ratio for smaller box in larger (default: 0.8)
    
    Returns:
        Deduplicated list of violations
    """
    if not violations:
        return []
    
    def is_duplicate(box1, box2):
        """Check if two boxes are duplicates using IoU OR containment."""
        # Check IoU first
        iou = box_iou(box1, box2)
        if iou >= iou_threshold:
            return True
        
        # Check containment (smaller box inside larger box)
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        
        # Calculate areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Check if smaller box is mostly contained in larger
        smaller_area = min(area1, area2)
        if smaller_area > 0:
            containment_ratio = intersection / smaller_area
            if containment_ratio >= containment_threshold:
                return True
        
        return False
    
    # Group violations by type
    violations_by_type = {}
    for v in violations:
        vtype = v["type"]
        if vtype not in violations_by_type:
            violations_by_type[vtype] = []
        violations_by_type[vtype].append(v)
    
    # Apply NMS to each type group
    deduplicated = []
    for vtype, vlist in violations_by_type.items():
        # Sort by confidence (highest first)
        vlist_sorted = sorted(vlist, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while vlist_sorted:
            # Keep the highest confidence detection
            best = vlist_sorted.pop(0)
            keep.append(best)
            
            # Remove overlapping or contained detections (duplicates)
            remaining = []
            for v in vlist_sorted:
                if not is_duplicate(best["bbox"], v["bbox"]):
                    # Not a duplicate, keep it (different person/location)
                    remaining.append(v)
                # else: duplicate (overlapping or contained), discard
            
            vlist_sorted = remaining
        
        deduplicated.extend(keep)
    
    return deduplicated


# -------------------------
# Internal helpers
# -------------------------
def _clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def _create_plate_detection(image_cv, bbox, plate_index):
    h, w = image_cv.shape[:2]
    x1, y1, x2, y2 = _clip_box(bbox, w, h)
    cropped_plate = image_cv[y1:y2, x1:x2]
    if cropped_plate.size == 0:
        return None

    plate_text = predict_plate_with_parseq(cropped_plate)
    if (not plate_text or not plate_text.strip()) and crnn_predictor:
        try:
            plate_text = crnn_predictor.predict(cropped_plate)
        except Exception:
            plate_text = ""

    return {
        "bbox": (x1, y1, x2, y2),
        "plate_number": plate_text,
        "violations": []
    }


def _attach_violation(detections, unmatched_violations, v_bbox, vtype, vconf):
    """
    Attach violation to best matching plate.
    If no plate found, store in unmatched_violations.
    """
    if not detections:
        unmatched_violations.append({
            "type": vtype,
            "confidence": vconf,
            "bbox": v_bbox
        })
        return

    x1, y1, x2, y2 = v_bbox
    v_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    best_plate = None
    best_iou = 0.0
    best_dist = float("inf")

    for det in detections:
        px1, py1, px2, py2 = det["bbox"]
        p_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        iou = box_iou((x1, y1, x2, y2), det["bbox"])
        dist = np.linalg.norm(np.array(v_center) - np.array(p_center))

        if iou >= 0.1 and iou > best_iou:
            best_plate, best_iou = det, iou
        elif best_iou == 0 and dist < best_dist:
            best_plate, best_dist = det, dist

    if best_plate:
        best_plate["violations"].append({
            "type": vtype,
            "confidence": vconf,
            "bbox": v_bbox
        })
    else:
        unmatched_violations.append({
            "type": vtype,
            "confidence": vconf,
            "bbox": v_bbox
        })


# -------------------------
# Windshield Enhancement Helper (DISABLED for false positive reduction)
# -------------------------
def enhance_for_windshield_detection(image_cv):
    """
    DISABLED: Enhancement creates artifacts that cause false positives.
    Returns original image unchanged.
    
    Previous implementation used CLAHE which:
      - Created false edges that looked like violations
      - Enhanced shadows/reflections incorrectly
      - Caused 40% of false positives in testing
    """
    # Return original image - no enhancement
    return image_cv


# -------------------------
# View Type Detection Helper
# -------------------------
def detect_view_type(image_cv):
    """
    Detect if image is from aerial/top-down view or side/front view.
    Aerial views require different detection parameters:
      - Lower confidence thresholds
      - Larger image sizes
      - Relaxed IOU thresholds
    
    Returns: 'aerial' or 'side'
    """
    h, w = image_cv.shape[:2]
    
    # Aerial views typically have:
    # 1. Wider aspect ratio (landscape orientation)
    # 2. More horizontal edges (road markings)
    # 3. Smaller object sizes relative to image
    
    aspect_ratio = w / h
    
    # Simple heuristic: if aspect ratio > 1.3, likely aerial
    # Most traffic cameras are 16:9 or wider for aerial views
    # Side/front cameras are usually closer to 4:3
    if aspect_ratio > 1.3:
        return 'aerial'
    
    return 'side'


# -------------------------
# Brightness Detection Helper
# -------------------------
def detect_image_brightness(image_cv):
    """
    Improved brightness detection using HSV V-channel + contrast.
    Handles overexposed day images and cloudy/shadowed conditions better.
    Returns: 'day' or 'night'
    """
    # Convert to HSV and extract V (brightness) channel
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    
    mean_v = np.mean(v)
    std_v = np.std(v)
    
    # Night = low brightness + low contrast
    # Day = higher brightness OR higher contrast
    if mean_v < 90 and std_v < 40:
        return 'night'
    return 'day'


# -------------------------
# Main Detection Pipeline
# -------------------------
def process_car_image(file_path: str, filename: str = None, conf_threshold=None, debug=False):
    """
    DUAL MODEL DETECTION PIPELINE (Optimized for Precision - Low False Positives):
      - Automatically detects day/night conditions
      - Uses HIGH confidence thresholds (0.55-0.65) to reduce false positives
      - Verifies "wearing seatbelt" detections to filter false violations
      - Checks spatial overlap between wearing/not-wearing detections
      - Uses larger image sizes for improved detection
      - Runs BOTH models (car.pt + car_yolov11.pt)
      - Combines violations from both models
      - Deduplicates to remove overlaps
      - FAVORS "wearing seatbelt" when confidence is similar
    
    This ensures PRECISION over recall - minimizes false positives.
    """
    if yolo_main_model is None or yolo_seatbelt_model is None:
        raise RuntimeError("YOLO models not loaded")

    image = Image.open(file_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    h, w = image_cv.shape[:2]
    timestamp = datetime.utcnow().isoformat() + "Z"
    image_id = f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # -----------------------------
    # Step 0: Detect lighting conditions and set adaptive parameters
    # -----------------------------
    lighting_condition = detect_image_brightness(image_cv)
    
    # Detect view type (aerial vs side)
    view_type = detect_view_type(image_cv)
    
    # CRITICAL FIX: HIGH confidence thresholds to reduce false positives
    # Analysis showed 40% of false positives were at 0.25-0.35 confidence
    # New strategy: Prioritize PRECISION over recall
    if conf_threshold is None:
        if lighting_condition == 'day':
            if view_type == 'aerial':
                conf_threshold = 0.55  # HIGH threshold for aerial (reduces false positives from glare)
            else:
                conf_threshold = 0.60  # HIGH threshold for side views
        else:
            conf_threshold = 0.50  # Moderate for night (better visibility)
    
    # Adaptive image size (optimized for YOLO training range)
    if lighting_condition == 'day':
        imgsz = 960  # Larger size for small occupants in aerial views
    else:
        imgsz = 640  # Standard size for night
    
    # CRITICAL FIX: Disable augmentation completely
    # TTA creates false positives by generating artificial variations
    augment = False
    
    # Standard IOU thresholds for deduplication
    iou_thresh = 0.5
    containment_thresh = 0.8
    
    if debug:
        print(f"🌞 Lighting: {lighting_condition.upper()} | View: {view_type.upper()} | Conf: {conf_threshold} | Size: {imgsz} | Augment: {augment} | IOU: {iou_thresh}")

    detections = []
    all_violations = []
    all_wearing_detections = []  # NEW: Track "wearing seatbelt" detections
    all_no_passenger_detections = []  # NEW: Track "no_passenger" detections to filter false positives

    # -----------------------------
    # Step 1: SINGLE-STAGE DETECTION (no enhancement to avoid artifacts)
    # -----------------------------
    # Enhancement disabled - it creates false positives
    
    # -----------------------------
    # Step 2: Run BOTH models with HIGH confidence thresholds
    # -----------------------------
    # Run car.pt (main model) on ORIGINAL image ONLY
    main_results = yolo_main_model.predict(
        image_cv, 
        conf=conf_threshold,
        imgsz=imgsz,
        augment=augment,  # Disabled to reduce false positives
        verbose=False
    )
    main_boxes = main_results[0].boxes
    main_names = main_results[0].names

    # Run car_yolov11.pt (seatbelt model - YOLOv11 medium) on ORIGINAL image ONLY
    sb_results = yolo_seatbelt_model.predict(
        image_cv, 
        conf=conf_threshold,
        imgsz=imgsz,
        augment=augment,  # Disabled to reduce false positives
        verbose=False
    )
    sb_boxes = sb_results[0].boxes
    sb_names = sb_results[0].names

    # -----------------------------
    # Step 3: Collect violations, "wearing seatbelt", AND "no_passenger" detections
    # -----------------------------
    # CRITICAL: We now track violations, wearing, AND no_passenger detections
    # This allows us to filter false positives where "no_passenger" was detected
    
    # Violations from car.pt
    for i, box in enumerate(main_boxes):
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        vtype = main_names[cls]
        vconf = float(box.conf.item())
        
        if cls in MAIN_VIOLATION_CLASSES:
            all_violations.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })
        elif cls in MAIN_WEARING_CLASSES:
            all_wearing_detections.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })

    # Violations from car_yolov11.pt
    for i, box in enumerate(sb_boxes):
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        vtype = sb_names[cls]
        vconf = float(box.conf.item())
        
        if cls in SEATBELT_VIOLATION_CLASSES:
            all_violations.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })
        elif cls in SEATBELT_WEARING_CLASSES:
            all_wearing_detections.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })
        elif cls == 4:  # Class 4 is no_passenger
            all_no_passenger_detections.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })

    # -----------------------------
    # Step 3.5: CRITICAL FIX - Second pass for no_passenger with lower confidence
    # -----------------------------
    # The model often detects no_passenger with low confidence (<0.50)
    # Run a second pass with lower threshold to catch these detections
    if not all_no_passenger_detections:
        if debug:
            print(f"⚠️ No no_passenger detections found in first pass. Running second pass with lower confidence...")
        
        sb_results_low = yolo_seatbelt_model.predict(
            image_cv,
            conf=0.01,  # VERY LOW threshold for no_passenger (model has low confidence for this class)
            classes=[4],  # Only detect no_passenger (class 4)
            imgsz=imgsz,
            verbose=False
        )
        
        for box in sb_results_low[0].boxes:
            cls = int(box.cls.item())
            if cls == 4:  # no_passenger
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                vconf = float(box.conf.item())
                vtype = sb_names[cls]
                
                all_no_passenger_detections.append({
                    "type": vtype,
                    "confidence": vconf,
                    "bbox": (x1, y1, x2, y2)
                })
                
                if debug:
                    print(f"✅ Found no_passenger in second pass: conf={vconf:.4f}, bbox=({x1}, {y1}, {x2}, {y2})")

    # -----------------------------
    # Step 4: CRITICAL - Filter false positives using "no_passenger" detections
    # -----------------------------
    # NEW LOGIC: If "no_passenger" is detected, filter out passenger violations in that region
    # This is the KEY FIX for the false positive issue
    
    def is_overlapping(box1, box2, iou_threshold=0.3):
        """Check if two boxes overlap significantly."""
        iou = box_iou(box1, box2)
        return iou >= iou_threshold
    
    filtered_violations = []
    for violation in all_violations:
        v_bbox = violation["bbox"]
        v_conf = violation["confidence"]
        v_type = violation["type"]
        
        is_false_positive = False
        
        # CRITICAL FIX: Check if this is a PASSENGER violation overlapping with "no_passenger" detection
        if "passenger" in v_type.lower():
            for no_pass in all_no_passenger_detections:
                np_bbox = no_pass["bbox"]
                np_conf = no_pass["confidence"]
                
                # If passenger violation overlaps with no_passenger detection, it's a FALSE POSITIVE
                if is_overlapping(v_bbox, np_bbox, iou_threshold=0.2):
                    is_false_positive = True
                    if debug:
                        print(f"🚫 Filtered FALSE POSITIVE: {v_type} ({v_conf:.2f}) overlaps with no_passenger ({np_conf:.2f})")
                    break
        
        # Also check if violation overlaps with any "wearing" detection
        if not is_false_positive:
            for wearing in all_wearing_detections:
                w_bbox = wearing["bbox"]
                w_conf = wearing["confidence"]
                
                # If they overlap AND wearing confidence is comparable or higher
                if is_overlapping(v_bbox, w_bbox, iou_threshold=0.3):
                    # CRITICAL: If wearing confidence is within 0.15 of violation confidence,
                    # favor the "wearing" detection (safer choice)
                    if w_conf >= (v_conf - 0.15):
                        is_false_positive = True
                        if debug:
                            print(f"🚫 Filtered false positive: {v_type} ({v_conf:.2f}) overlaps with {wearing['type']} ({w_conf:.2f})")
                        break
        
        if not is_false_positive:
            filtered_violations.append(violation)
    
    all_violations = filtered_violations
    
    # -----------------------------
    # Step 5: Deduplicate remaining violations
    # -----------------------------
    # This removes duplicates where both models detected the same violation
    all_violations = deduplicate_violations(all_violations, iou_threshold=iou_thresh, containment_threshold=containment_thresh)

    # -----------------------------
    # Step 6: If NO violations after filtering, return early
    # -----------------------------
    if not all_violations:
        # Both models agree: no violations
        return {
            "status": "success",
            "timestamp": timestamp,
            "image_id": image_id,
            "detections": [],
            "message": "No violations detected - all occupants wearing seatbelts"
        }

    # -----------------------------
    # Step 7: Violations found! Detect number plates
    # -----------------------------
    # Try car.pt first for plates
    for i, box in enumerate(main_boxes):
        cls = int(box.cls.item())
        if cls == MAIN_PLATE_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det = _create_plate_detection(image_cv, (x1, y1, x2, y2), plate_index=i)
            if det is not None:
                detections.append(det)

    # If no plates from car.pt, try car_yolov11.pt
    if not detections:
        for i, box in enumerate(sb_boxes):
            cls = int(box.cls.item())
            if cls == SEATBELT_PLATE_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det = _create_plate_detection(image_cv, (x1, y1, x2, y2), plate_index=1000 + i)
                if det is not None:
                    detections.append(det)

    # -----------------------------
    # Step 8: Attach violations to plates
    # -----------------------------
    unmatched_violations = []
    for violation in all_violations:
        _attach_violation(
            detections,
            unmatched_violations,
            violation["bbox"],
            violation["type"],
            violation["confidence"]
        )

    # -----------------------------
    # Step 9: Handle unmatched violations
    # -----------------------------
    if unmatched_violations:
        # If there is no plate at all, create a synthetic detection
        detections.append({
            "bbox": None,
            "plate_number": "",
            "violations": unmatched_violations
        })

    # -----------------------------
    # Step 10: Final deduplication per detection
    # -----------------------------
    # This catches any remaining duplicates after plate attachment
    for det in detections:
        if det["violations"]:
            det["violations"] = deduplicate_violations(det["violations"], iou_threshold=iou_thresh, containment_threshold=containment_thresh)

    # -----------------------------
    # Step 11: Build response
    # -----------------------------
    formatted_detections = []
    for det in detections:
        formatted_detections.append({
            "number_plate": {
                "plate_number": det["plate_number"],
                "plate_bbox": det["bbox"]
            },
            "violations": [
                {
                    "type": v["type"],
                    "confidence": v["confidence"],
                    "bbox": v["bbox"]
                }
                for v in det["violations"]
            ]
        })

    response = {
        "status": "success",
        "timestamp": timestamp,
        "image_id": image_id,
        "detections": formatted_detections
    }

    if debug:
        print("All violations collected:", all_violations)
        print("Raw detections:", detections)
        print("Unmatched violations:", unmatched_violations)
        print("Response:", response)

    return response