# detector_yolov11_bike.py
# Bike violation detector using bike_model_new.pt (YOLOv11)
import os
import uuid
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
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
BIKE_MODEL_PATH = "models/bike.pt"  # New YOLOv11 bike model

CROPPED_PLATES_DIR = "bike_folders/cropped_number_plates"
OFFENSES_DIR = "bike_folders/bikes_with_offence"
NO_OFFENSES_DIR = "bike_folders/images_not_with_offence"

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
# Load YOLO model
# -------------------------
try:
    bike_model = YOLO(BIKE_MODEL_PATH)
    print(f"✅ Bike model loaded: {BIKE_MODEL_PATH}")
except Exception as e:
    print(f"❌ YOLO loading error: {e}")
    bike_model = None


# -------------------------
# Class mappings for bike_model_new.pt
# -------------------------
# Class 0: Number_plate
# Class 1: mobile_usage
# Class 2: pillion_not_wearing_helmet
# Class 3: pillion_wearing_helmet
# Class 4: rider_not_wearing_helmet
# Class 5: rider_wearing_helmet
# Class 6: triple_riding
# Class 7: vehicle

PLATE_CLASS = 0
VIOLATION_CLASSES = [1, 2, 4, 6]  # mobile_usage, pillion_not_wearing_helmet, rider_not_wearing_helmet, triple_riding
WEARING_CLASSES = [3, 5]  # pillion_wearing_helmet, rider_wearing_helmet
IGNORE_CLASSES = [7]  # vehicle (used for context only)


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


def resolve_conflicting_detections(violations, wearing_detections, iou_threshold=0.3):
    """
    Resolve conflicting detections where both "wearing" and "not wearing" are detected
    for the same person. Keep only the detection with higher confidence.
    
    Conflicts:
    - pillion_not_wearing_helmet vs pillion_wearing_helmet
    - rider_not_wearing_helmet vs rider_wearing_helmet
    
    Args:
        violations: List of violation detections (not wearing) with 'type', 'confidence', 'bbox'
        wearing_detections: List of wearing detections with 'type', 'confidence', 'bbox'
        iou_threshold: IoU threshold for considering detections as conflicting
    
    Returns:
        Tuple of (filtered_violations, filtered_wearing_detections)
    """
    # Map conflicting pairs by type name (check both exact match and substring)
    # These are the conflicting detection types that can't both be true
    conflict_pairs = {
        # Key: violation type (not wearing), Value: corresponding wearing type
        'pillion_not_wearing_helmet': 'pillion_wearing_helmet',
        'pillion_rider_not_wearing_helmet': 'pillion_wearing_helmet',  # Alternative naming
        'rider_not_wearing_helmet': 'rider_wearing_helmet',
        'rider_and_pillion_not_wearing_helmet': 'pillion_wearing_helmet'  # Also conflicts with pillion_wearing
    }
    
    # Group wearing detections by type for quick lookup
    wearing_by_type = {}
    for wearing in wearing_detections:
        w_type = wearing['type']
        if w_type not in wearing_by_type:
            wearing_by_type[w_type] = []
        wearing_by_type[w_type].append(wearing)
    
    # Process violations and check for conflicts
    filtered_violations = []
    wearing_to_remove = set()  # Track wearing detections to remove by their index/id
    
    for violation in violations:
        v_type = violation['type']
        v_bbox = violation['bbox']
        v_conf = violation['confidence']
        
        # Check if this violation has a conflicting wearing detection
        conflicting_wearing_type = None
        v_type_lower = v_type.lower()
        
        # Check for exact matches first
        if v_type in conflict_pairs:
            conflicting_wearing_type = conflict_pairs[v_type]
        else:
            # Check for substring matches - handle variations in naming
            # Check if violation contains key words that indicate a conflict
            if 'pillion' in v_type_lower and 'not_wearing' in v_type_lower:
                conflicting_wearing_type = 'pillion_wearing_helmet'
            elif 'rider' in v_type_lower and 'not_wearing' in v_type_lower and 'pillion' not in v_type_lower:
                # Only rider, not both rider and pillion
                conflicting_wearing_type = 'rider_wearing_helmet'
        
        if conflicting_wearing_type is None:
            # No conflict possible, keep violation
            filtered_violations.append(violation)
            continue
        
        # Check if there are any wearing detections of the conflicting type
        if conflicting_wearing_type not in wearing_by_type:
            # No conflicting wearing detection, keep violation
            filtered_violations.append(violation)
            continue
        
        # Check for overlapping wearing detections
        conflict_found = False
        for wearing in wearing_by_type[conflicting_wearing_type]:
            w_bbox = wearing['bbox']
            w_conf = wearing['confidence']
            
            # Check if they overlap significantly
            iou = box_iou(v_bbox, w_bbox)
            if iou >= iou_threshold:
                conflict_found = True
                
                # Keep the one with higher confidence
                if v_conf > w_conf:
                    # Keep violation, mark wearing for removal
                    wearing_to_remove.add(id(wearing))
                    filtered_violations.append(violation)
                else:
                    # Keep wearing, skip violation (don't add to filtered_violations)
                    pass
                break
        
        # If no conflict found, keep the violation
        if not conflict_found:
            filtered_violations.append(violation)
    
    # Remove conflicting wearing detections
    filtered_wearing = [w for w in wearing_detections if id(w) not in wearing_to_remove]
    
    return filtered_violations, filtered_wearing


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


def is_inside_vehicle(obj_bbox, vehicle_bbox, threshold=0.3):
    """
    Check if object is inside vehicle region.
    
    Args:
        obj_bbox: Object bounding box (x1, y1, x2, y2)
        vehicle_bbox: Vehicle bounding box (x1, y1, x2, y2)
        threshold: Minimum containment ratio (default: 0.3 = 30%)
    
    Returns:
        True if object is inside vehicle (containment >= threshold)
    """
    x1_i = max(obj_bbox[0], vehicle_bbox[0])
    y1_i = max(obj_bbox[1], vehicle_bbox[1])
    x2_i = min(obj_bbox[2], vehicle_bbox[2])
    y2_i = min(obj_bbox[3], vehicle_bbox[3])
    
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
    
    if obj_area == 0:
        return False
    
    containment_ratio = intersection / obj_area
    return containment_ratio >= threshold


def group_by_vehicle(vehicles, plates, violations, compliance):
    """
    Group plates, violations, and compliance by vehicle using spatial containment.
    
    This solves the cross-bike assignment problem in aerial/top-down camera views
    by using vehicle bounding boxes as spatial boundaries.
    
    Args:
        vehicles: List of vehicle detections with bbox
        plates: List of plate detections with bbox
        violations: List of violation detections with bbox, type, confidence
        compliance: List of compliance detections with bbox, type, confidence
    
    Returns:
        List of vehicle groups, each containing:
        - vehicle_bbox: Vehicle bounding box
        - plates: Plates inside this vehicle
        - violations: Violations inside this vehicle
        - compliance: Compliance inside this vehicle
    """
    if not vehicles:
        # No vehicles detected - return single group with all objects
        return [{
            "vehicle_bbox": None,
            "plates": plates,
            "violations": violations,
            "compliance": compliance
        }]
    
    vehicle_groups = []
    
    for vehicle in vehicles:
        v_bbox = vehicle["bbox"]
        
        # Find objects inside this vehicle using containment check
        vehicle_plates = [p for p in plates if is_inside_vehicle(p["bbox"], v_bbox)]
        vehicle_violations = [v for v in violations if is_inside_vehicle(v["bbox"], v_bbox)]
        vehicle_compliance = [c for c in compliance if is_inside_vehicle(c["bbox"], v_bbox)]
        
        # Only create group if it has at least a plate or violations/compliance
        if vehicle_plates or vehicle_violations or vehicle_compliance:
            vehicle_groups.append({
                "vehicle_bbox": v_bbox,
                "plates": vehicle_plates,
                "violations": vehicle_violations,
                "compliance": vehicle_compliance
            })
    
    # Handle objects not inside any vehicle (fallback)
    unmatched_plates = [p for p in plates if not any(is_inside_vehicle(p["bbox"], v["bbox"]) for v in vehicles)]
    unmatched_violations = [v for v in violations if not any(is_inside_vehicle(v["bbox"], veh["bbox"]) for veh in vehicles)]
    unmatched_compliance = [c for c in compliance if not any(is_inside_vehicle(c["bbox"], v["bbox"]) for v in vehicles)]
    
    if unmatched_plates or unmatched_violations or unmatched_compliance:
        vehicle_groups.append({
            "vehicle_bbox": None,
            "plates": unmatched_plates,
            "violations": unmatched_violations,
            "compliance": unmatched_compliance
        })
    
    return vehicle_groups


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
def process_bike_image(file_path: str, filename: str = None, conf_threshold=None, debug=False):
    """
    BIKE VIOLATION DETECTION PIPELINE (YOLOv11 - VALIDATION MODE):
      - Automatically detects day/night conditions
      - Uses LOW confidence thresholds (0.25) to see ALL detections
      - FALSE POSITIVE FILTERING DISABLED - shows raw model output
      - Deduplicates to remove overlaps
    
    VALIDATION MODE: Shows all detections for manual validation.
    Use this to verify model performance before applying filters.
    """
    if bike_model is None:
        raise RuntimeError("Bike YOLO model not loaded")

    image = Image.open(file_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    h, w = image_cv.shape[:2]
    timestamp = datetime.utcnow().isoformat() + "Z"
    image_id = f"BIKE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # -----------------------------
    # Step 0: Detect lighting conditions and set adaptive parameters
    # -----------------------------
    lighting_condition = detect_image_brightness(image_cv)
    
    # VALIDATION MODE: Lower confidence thresholds to see ALL detections
    if conf_threshold is None:
        if lighting_condition == 'day':
            conf_threshold = 0.25  # LOW threshold for validation (see all detections)
        else:
            conf_threshold = 0.25  # LOW threshold for validation
    
    # Adaptive image size
    if lighting_condition == 'day':
        imgsz = 960  # Larger size for better detection
    else:
        imgsz = 640  # Standard size for night
    
    # CRITICAL FIX: Disable augmentation completely
    augment = False
    
    # Standard IOU thresholds for deduplication
    iou_thresh = 0.5
    containment_thresh = 0.8
    
    if debug:
        print(f"🌞 Lighting: {lighting_condition.upper()} | Conf: {conf_threshold} | Size: {imgsz} | Augment: {augment}")

    detections = []
    all_violations = []
    all_wearing_detections = []  # Track "wearing helmet" detections

    # -----------------------------
    # Step 1: Run bike model
    # -----------------------------
    results = bike_model.predict(
        image_cv, 
        conf=conf_threshold,
        imgsz=imgsz,
        augment=augment,
        verbose=False
    )
    boxes = results[0].boxes
    names = results[0].names

    # -----------------------------
    # Step 2: Collect ALL detections (violations, compliance, vehicles, plates)
    # -----------------------------
    all_vehicles = []  # NEW: Track vehicle detections for spatial grouping
    all_plates = []    # NEW: Track plate detections separately
    
    for i, box in enumerate(boxes):
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        vtype = names[cls]
        vconf = float(box.conf.item())
        
        if cls in VIOLATION_CLASSES:
            all_violations.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })
        elif cls in WEARING_CLASSES:
            all_wearing_detections.append({
                "type": vtype,
                "confidence": vconf,
                "bbox": (x1, y1, x2, y2)
            })
        elif cls == PLATE_CLASS:
            all_plates.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": vconf
            })
        elif cls == 7:  # vehicle class
            all_vehicles.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": vconf
            })

    # -----------------------------
    # Step 3: FALSE POSITIVE FILTERING DISABLED FOR VALIDATION
    # -----------------------------
    # VALIDATION MODE: Skip false positive filtering to see ALL raw detections
    # This allows manual validation of model output before applying filters
    
    # def is_overlapping(box1, box2, iou_threshold=0.3):
    #     """Check if two boxes overlap significantly."""
    #     iou = box_iou(box1, box2)
    #     return iou >= iou_threshold
    # 
    # filtered_violations = []
    # for violation in all_violations:
    #     v_bbox = violation["bbox"]
    #     v_conf = violation["confidence"]
    #     v_type = violation["type"]
    #     
    #     # Check if this violation overlaps with any "wearing" detection
    #     is_false_positive = False
    #     for wearing in all_wearing_detections:
    #         w_bbox = wearing["bbox"]
    #         w_conf = wearing["confidence"]
    #         
    #         # If they overlap AND wearing confidence is comparable or higher
    #         if is_overlapping(v_bbox, w_bbox, iou_threshold=0.3):
    #             # CRITICAL: If wearing confidence is within 0.15 of violation confidence,
    #             # favor the "wearing" detection (safer choice)
    #             if w_conf >= (v_conf - 0.15):
    #                 is_false_positive = True
    #                 if debug:
    #                     print(f"🚫 Filtered false positive: {v_type} ({v_conf:.2f}) overlaps with {wearing['type']} ({w_conf:.2f})")
    #                 break
    #     
    #     if not is_false_positive:
    #         filtered_violations.append(violation)
    # 
    # all_violations = filtered_violations
    
    # VALIDATION MODE: Keep all violations without filtering
    if debug:
        print(f"📊 Raw violations (no filtering): {len(all_violations)}")
        print(f"📊 Wearing detections: {len(all_wearing_detections)}")
    
    # -----------------------------
    # Step 3.5: Resolve conflicting detections (wearing vs not wearing)
    # -----------------------------
    # If both "wearing" and "not wearing" are detected for same person, keep higher confidence
    all_violations, all_wearing_detections = resolve_conflicting_detections(
        all_violations, 
        all_wearing_detections, 
        iou_threshold=0.3
    )
    
    if debug:
        print(f"📊 After conflict resolution - Violations: {len(all_violations)}, Wearing: {len(all_wearing_detections)}")
    
    # -----------------------------
    # Step 4: Deduplicate remaining violations
    # -----------------------------
    all_violations = deduplicate_violations(all_violations, iou_threshold=iou_thresh, containment_threshold=containment_thresh)

    # -----------------------------
    # Step 5: If NO violations after filtering, return early
    # -----------------------------
    if not all_violations:
        return {
            "status": "success",
            "timestamp": timestamp,
            "image_id": image_id,
            "detections": [],
            "message": "No violations detected - all riders wearing helmets properly"
        }

    # -----------------------------
    # Step 6: Group detections by vehicle (SPATIAL GROUPING - NEW!)
    # -----------------------------
    # This solves cross-bike assignment in aerial views by using vehicle boundaries
    vehicle_groups = group_by_vehicle(all_vehicles, all_plates, all_violations, all_wearing_detections)
    
    if debug:
        print(f"🚗 Vehicle groups: {len(vehicle_groups)}")
        for idx, group in enumerate(vehicle_groups):
            print(f"  Group {idx+1}: {len(group['plates'])} plates, {len(group['violations'])} violations, {len(group['compliance'])} compliance")
    
    # -----------------------------
    # Step 7: Create detections from vehicle groups
    # -----------------------------
    for group in vehicle_groups:
        # Get plate for this vehicle group
        if group["plates"]:
            # Use first plate (or best plate if multiple)
            plate_info = group["plates"][0]
            det = _create_plate_detection(image_cv, plate_info["bbox"], plate_index=0)
            
            if det is not None:
                # Add violations and compliance from this vehicle group
                det["violations"] = group["violations"]
                det["compliance"] = group["compliance"]
                detections.append(det)
        else:
            # No plate detected for this vehicle, but has violations/compliance
            if group["violations"] or group["compliance"]:
                detections.append({
                    "bbox": None,
                    "plate_number": "",
                    "violations": group["violations"],
                    "compliance": group["compliance"]
                })

    # -----------------------------
    # Step 9: Final deduplication per detection
    # -----------------------------
    for det in detections:
        if "violations" not in det:
            det["violations"] = []
        if "compliance" not in det:
            det["compliance"] = []
            
        if det["violations"]:
            det["violations"] = deduplicate_violations(det["violations"], iou_threshold=iou_thresh, containment_threshold=containment_thresh)
        
        if det["compliance"]:
            det["compliance"] = deduplicate_violations(det["compliance"], iou_threshold=iou_thresh, containment_threshold=containment_thresh)

    # -----------------------------
    # Step 10: Build response with violations AND compliance
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
                for v in det.get("violations", [])
            ],
            "compliance": [
                {
                    "type": c["type"],
                    "confidence": c["confidence"],
                    "bbox": c["bbox"]
                }
                for c in det.get("compliance", [])
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
        print("All compliance collected:", all_wearing_detections)
        print("All vehicles detected:", all_vehicles)
        print("Vehicle groups created:", len(vehicle_groups))
        print("Raw detections:", detections)
        print("Response:", response)

    return response


def process_bike_image_with_annotation(file_path: str, filename: str = None, conf_threshold=0.25):
    """
    Process bike image and return annotated image with vehicle grouping.
    Uses bike.pt model to create a visual annotation.
    
    Args:
        file_path: Path to the image file
        filename: Original filename (optional)
        conf_threshold: Confidence threshold for detection (default: 0.25)
    
    Returns:
        numpy array (RGB) of annotated image
    """
    if bike_model is None:
        raise RuntimeError("Bike YOLO model not loaded")
    
    # Load image
    image = Image.open(file_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image_cv.shape[:2]
    
    # Run detection
    results = bike_model.predict(image_cv, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    
    # Class names mapping
    CLASS_NAMES = {
        0: 'Number_plate',
        1: 'mobile_usage',
        2: 'pillion_not_wearing_helmet',
        3: 'pillion_wearing_helmet',
        4: 'rider_not_wearing_helmet',
        5: 'rider_wearing_helmet',
        6: 'triple_riding',
        7: 'vehicle'
    }
    
    # Color mapping for each class (BGR format)
    CLASS_COLORS = {
        0: (255, 255, 0),    # Cyan - Number_plate
        1: (0, 0, 255),      # Red - mobile_usage
        2: (0, 0, 139),      # Dark Red - pillion_not_wearing_helmet
        3: (0, 255, 0),      # Green - pillion_wearing_helmet
        4: (0, 69, 255),     # Orange - rider_not_wearing_helmet
        5: (255, 0, 255),    # Magenta - rider_wearing_helmet
        6: (255, 255, 255),  # White - triple_riding
        7: (255, 255, 255)   # White - vehicle
    }
    
    # Collect all detections
    detections = []
    vehicles = []
    other_objects = []
    
    for box in boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf.item())
        class_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        
        det = {
            'class_id': cls_id,
            'class_name': class_name,
            'bbox': (x1, y1, x2, y2),
            'confidence': conf
        }
        detections.append(det)
        
        if cls_id == 7:  # vehicle
            vehicles.append(det)
        else:
            other_objects.append(det)
    
    # Group detections by vehicle
    vehicle_groups = []
    if vehicles:
        for vehicle in vehicles:
            vehicle_groups.append({
                'bbox': vehicle['bbox'],
                'confidence': vehicle['confidence'],
                'objects': []
            })
        
        # Associate objects with vehicles
        for obj in other_objects:
            obj_bbox = obj['bbox']
            best_vehicle = None
            best_score = 0
            
            for vehicle_group in vehicle_groups:
                v_bbox = vehicle_group['bbox']
                iou = box_iou(obj_bbox, v_bbox)
                containment = is_inside_vehicle(obj_bbox, v_bbox, threshold=0.3)
                score = max(iou, 1.0 if containment else 0.0)
                
                if score > best_score:
                    best_score = score
                    best_vehicle = vehicle_group
            
            if best_vehicle and best_score > 0.1:
                best_vehicle['objects'].append(obj)
            else:
                # Add to first group or create new group
                if vehicle_groups:
                    vehicle_groups[0]['objects'].append(obj)
    else:
        # No vehicles detected, create single group
        vehicle_groups = [{
            'bbox': None,
            'confidence': None,
            'objects': other_objects
        }]
    
    # Create visualization
    vis_image = image_cv.copy()
    
    # Draw each vehicle group
    for idx, group in enumerate(vehicle_groups):
        # Draw vehicle bbox if exists
        if group.get('bbox') is not None:
            v_bbox = group['bbox']
            color = CLASS_COLORS.get(7, (255, 255, 255))
            cv2.rectangle(vis_image, (v_bbox[0], v_bbox[1]), (v_bbox[2], v_bbox[3]), color, 3)
            conf = group.get('confidence', 0.0) if group.get('confidence') is not None else 0.0
            label = f"Vehicle #{idx+1} ({conf:.2f})"
            cv2.putText(vis_image, label, (v_bbox[0], v_bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw associated objects
        for obj in group.get('objects', []):
            cls_id = obj['class_id']
            cls_name = obj['class_name']
            bbox = obj['bbox']
            conf = obj['confidence']
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{cls_name} ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for label
            cv2.rectangle(vis_image, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # Text
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis_image, "Class Legend:", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25
    
    for cls_id, cls_name in CLASS_NAMES.items():
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(vis_image, (10, legend_y - 15), (30, legend_y), color, -1)
        cv2.putText(vis_image, cls_name, (35, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
    
    # Convert BGR to RGB for proper display
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    return vis_image_rgb
