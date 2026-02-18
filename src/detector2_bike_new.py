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
# Valid Indian state/UT codes (first 2 characters of number plate)
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
    'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
    'TN', 'TS', 'TR', 'UK', 'UP', 'WB'
}

# Strict Regex: State code MUST be valid to prevent hallucinations like LB07
_states_str = "|".join(sorted(list(INDIAN_STATE_CODES)))
PLATE_REGEX = re.compile(rf'^({_states_str})[0-9]{{2}}[A-Z]{{1,2}}[0-9]{{4}}$')

CRNN_MODEL_PATH = "src/fine_tuned_crnn.pth"
BIKE_MODEL_PATH = "models/bike_yolov11_new.pt"  # New YOLOv11 bike model

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
# Verification Model (ResNet/MobileNet)
# -------------------------
# -------------------------
# Verification Model (ResNet/MobileNet)
# -------------------------
class VerificationModel:
    def __init__(self, model_path):
        import traceback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        
        try:
            from torchvision import models, transforms
            
            # --- Try loading ResNet50 (Since we confirmed the model is ResNet50) ---
            try:
                # 1. Initialize Architecture
                self.model = models.resnet50(pretrained=False)
                # 2. Match Last Layer
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, 2)
                
                # 3. Load Weights
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                # print("✅ [Verifier] Loaded ResNet50 model")
                
            except Exception as e_resnet:
                print(f"⚠️ [Verifier] ResNet50 load failed, trying MobileNetV3: {e_resnet}")
                # Fallback to MobileNetV3
                try:
                    self.model = models.mobilenet_v3_large(weights=None)
                    num_ftrs = self.model.classifier[3].in_features
                    self.model.classifier[3] = torch.nn.Linear(num_ftrs, 2)
                    
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device)
                    self.model.eval()
                    print("✅ [Verifier] Loaded MobileNetV3 model")
                except Exception as e_mobile:
                    print(f"❌ [Verifier] Both architectures failed: {e_mobile}")
                    self.model = None
                    return
            
            # --- Define Transform ---
            # IMPORTANT: Must match training logic!
            # Training used: RandomResizedCrop(128)
            # Inference should use: Resize(160) -> CenterCrop(128)
            self.transform = transforms.Compose([
                transforms.ToPILImage(), # Converts Tensor/Array to PIL Image
                transforms.Resize(160),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print(f"✅ Loaded violation verifier: {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load verifier: {e}")
            traceback.print_exc()
            self.model = None

    def verify(self, image_crop):
        """
        Verify if a crop is actually a violation.
        Input: BGR image crop (numpy array)
        Returns: 'violation', 'safe', or 'error'
        """
        if self.model is None:
            return 'violation' # Fail-safe: Assume violation if verifier is broken
            
        try:
            if image_crop.size == 0:
                print("⚠️ [Verifier] Empty crop received")
                return 'error'
            
            # 1. Convert BGR to RGB (Model expects RGB)
            # Input is likely BGR from cv2.imread
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
                
            # 2. Preprocess
            # transform takes PIL Image or Tensor. ToPILImage handles ndarray.
            tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # 3. Predict
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)
            
            # Probabilities for debugging
            safe_prob = probs[0][0].item()
            violation_prob = probs[0][1].item()
            
            idx = preds.item()
            confidence = conf.item()
            
            # Assuming alphabetical order for ImageFolder: 0=safe, 1=violation
            label = 'violation' if idx == 1 else 'safe'
            
            print(f"   🛡️ [Verifier] Safe: {safe_prob:.4f} | Violation: {violation_prob:.4f} -> {label.upper()}")
            
            return label
            
        except Exception as e:
            print(f"❌ [Verifier] Error during verification: {e}")
            return 'error'

# Initialize verifier
VERIFIER_PATH = "models/violation_verifier.pth"
verifier = VerificationModel(VERIFIER_PATH)


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


def enhance_image(img):
    # Upscale 2x
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def correct_plate_chars(text):
    if len(text) not in [9, 10]:
        return text
    
    chars = list(text)
    
    # Mapping for Digits -> Letters (for positions that MUST be letters)
    d2c = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', 
        '5': 'S', '6': 'G', '7': 'T', '8': 'B'
    }
    
    # Mapping for Letters -> Digits (for positions that MUST be digits)
    c2d = {
        'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 
        'J': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'
    }

    # Define masks: L=Letter, N=Number
    # 10 chars: LL NN LL NNNN (standard)
    if len(text) == 10:
        mask = "LLNNLLNNNN"
    # 9 chars: LL NN L NNNN
    else:
        mask = "LLNNLNNNN"
        
    for i, m in enumerate(mask):
        c = chars[i]
        if m == 'L':
            if c.isdigit() and c in d2c:
                print(f"   🛠️ Correction: Swapped '{c}' -> '{d2c[c]}' at pos {i}")
                chars[i] = d2c[c]
        elif m == 'N':
            if c.isalpha() and c in c2d:
                print(f"   🛠️ Correction: Swapped '{c}' -> '{c2d[c]}' at pos {i}")
                chars[i] = c2d[c]

    return "".join(chars)

def clean_plate_number(plate_text):
    """
    Clean OCR output by removing junk characters before valid Indian state code.
    Also applies character correction for common confused characters.
    """
    if not plate_text or len(plate_text) < 2:
        return plate_text
    
    # 1. Apply Smart Character Correction first if valid length
    if len(plate_text) in [9, 10]:
        plate_text = correct_plate_chars(plate_text)

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
        # Critical Fallback: Fuzzy State Correction
        # This handles common OCR confusions for first 2 letters (e.g., L->K, B->A)
        p2 = plate_text[:2]
        
        # 1. Fuzzy overrides (Hamming distance based or visual confusion)
        fuzzy_states = {
            'LB': 'KA', 'KB': 'KA', 'KH': 'KA', 'LA': 'KA', 'KZ': 'KA',
            'UB': 'WB', 'HB': 'HR', 'MA': 'MH', 'NA': 'KA',
            # Digit confusions for State Code
            '70': 'KA', '20': 'KA', '10': 'KA'
        }
        
        if p2 in fuzzy_states:
            new_state = fuzzy_states[p2]
            print(f"   ⚠️ OCR Correction: Fuzzy State Match '{p2}' -> '{new_state}'")
            return new_state + plate_text[2:]
            
        # 2. Heuristic: If 1st char matches, try correcting 2nd char to valid state
        # (Prioritize KA for this context)
        first_char = p2[0]
        possible_states = [s for s in INDIAN_STATE_CODES if s.startswith(first_char)]
        if possible_states:
            # Pick first valid (e.g., K -> KA, KL...)
            # Bias K -> KA
            if 'KA' in possible_states: return 'KA' + plate_text[2:]
            return possible_states[0] + plate_text[2:]
            
        # 3. Truncation Heuristic (Specific for KA data)
        # If text starts with 'A' and next is digit (e.g. A09...), assume 'KA'
        if len(plate_text) > 1 and plate_text[0] == 'A' and plate_text[1].isdigit():
             print(f"   ⚠️ OCR Correction: Fixed truncated 'A' -> 'KA'")
             return 'KA' + plate_text[1:]
             
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
    Returns cleaned uppercase text without spaces and confidence.
    """
    if model_parseq is None or transform_parseq is None:
        return "", 0.0
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = transform_parseq(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model_parseq(tensor)
        p = logits.softmax(-1)
        text, probs = model_parseq.tokenizer.decode(p)
        
        # Calculate confidence
        try:
            if isinstance(probs, list) and len(probs) > 0 and hasattr(probs[0], 'mean'):
                 conf = probs[0].mean().item()
            else:
                 conf = 0.0
        except Exception:
            conf = 0.0
            
        if not text:
            return "", 0.0
        return text[0].replace(" ", "").upper(), conf
    except Exception as e:
        return "", 0.0


def predict_plate_with_parseq(image_input):
    """
    Enhanced plate recognition with 2-line plate support, enhancement, and robust selection.
    """
    if model_parseq is None or transform_parseq is None:
        return ""
        
    # Load image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            return ""
    else:
        img = image_input

    # Helper to get candidates
    def get_candidates(input_img):
        cands = []
        import re
        clean = lambda s: re.sub(r'[^A-Z0-9]', '', s)
        
        # 1. Full
        full_raw, full_conf = ocr_line(input_img)
        full_text = clean(full_raw)
        cands.append({'text': full_text, 'conf': full_conf})
        
        # Add corrected version
        corr_full = clean_plate_number(full_text)
        if corr_full != full_text:
             cands.append({'text': corr_full, 'conf': full_conf})
        
        # 2. Split
        processed = remove_ind_strip(input_img)
        h = processed.shape[0]
        top, bottom = split_two_lines(processed)
        
        t1_raw, c1 = ocr_line(top)
        t2_raw, c2 = ocr_line(bottom)
        split_text = clean(t1_raw) + clean(t2_raw)
        split_conf = (c1 + c2) / 2 if (c1 > 0 or c2 > 0) else 0.0
        cands.append({'text': split_text, 'conf': split_conf})
        
        # Add corrected version
        corr_split = clean_plate_number(split_text)
        if corr_split != split_text:
             cands.append({'text': corr_split, 'conf': split_conf})
        
        return cands

    try:
        # Get initial candidates (Original)
        candidates = get_candidates(img)
        
        # Check if any satisfy regex
        has_match = any(bool(PLATE_REGEX.match(c['text'])) for c in candidates)
        
        # If no match, try Multi-Scale / Multi-Filter approach
        if not has_match:
            try:
                print(f"   🔍 ANPR: No Regex match. Applying Advanced Enhancements (Upscale, Erode, Contrast)...")
                
                # 1. Upscale + Sharpen (Good for blur)
                img_enhanced = enhance_image(img)
                enh_cands = get_candidates(img_enhanced)
                candidates.extend(enh_cands)
                
                # 2. Erosion (Good for bold/thick text - separates G from O)
                kernel = np.ones((2,2), np.uint8)
                img_eroded = cv2.erode(img, kernel, iterations=1)
                erode_cands = get_candidates(img_eroded)
                candidates.extend(erode_cands)

                # 3. Contrast (CLAHE) - (Good for shadows)
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_yuv[:,:,0])
                img_contrast = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                contrast_cands = get_candidates(img_contrast)
                candidates.extend(contrast_cands)

                # Log success
                if any(bool(PLATE_REGEX.match(c['text'])) for c in enh_cands + erode_cands + contrast_cands):
                    print(f"   ✅ ANPR: Advanced Enhancement found a valid match!")
            except Exception as e:
                print(f"   ⚠️ Enhancement loop error: {e}")
                pass
            
        # 4. Filter for 11-char plates with 5 digits at end (Common artifact)
        # e.g. KA03HU40292 -> KA03HU0292 (Drops extra digit)
        extra_digit_regex = re.compile(r'^([A-Z]{2}[0-9]{2}[A-Z]{1,2})([0-9]{5})$')
        new_cands = []
        for c in candidates:
            m = extra_digit_regex.match(c['text'])
            if m:
                prefix = m.group(1)
                digits = m.group(2)
                # Generate 5 variants by dropping one digit
                for i in range(5):
                    variant = prefix + digits[:i] + digits[i+1:]
                    new_cands.append({'text': variant, 'conf': c['conf']})
        
        if new_cands:
            print(f"   ⚠️ OCR Strategy: Generated {len(new_cands)} variants for 5-digit number plate")
            candidates.extend(new_cands)
            
        # Selection Logic
        # Priority: Valid Regex > Valid State Code > Length close to 10 > High Conf
        def sort_key(c):
            text = c['text']
            is_valid = bool(PLATE_REGEX.match(text))
            # New heuristic: Valid State Code Prefix (Helps pick KA... over A...)
            has_valid_state = text[:2] in INDIAN_STATE_CODES
            dist_to_10 = abs(len(text) - 10)
            conf = c['conf']
            # We want to maximize: (is_valid, has_valid_state, -dist, conf)
            return (is_valid, has_valid_state, -dist_to_10, conf)
            
        best_cand = max(candidates, key=sort_key)
        
        # Log final selection
        print(f"   🏁 ANPR Final Selection: {best_cand['text']} (Conf: {best_cand['conf']:.2f})")
        
        return best_cand['text']
        
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
# Class mappings for bike_yolov11_new.pt
# -------------------------
# Class 0: 6 (IGNORED)
# Class 1: Number_plate
# Class 2: mobile_usage
# Class 3: pillion_not_wearing_helmet
# Class 4: pillion_wearing_helmet
# Class 5: rider_not_wearing_helmet
# Class 6: rider_wearing_helmet
# Class 7: triple_riding
# Class 8: vehicle

IGNORE_CLASSES = [0]  # Class 0 = "6" (ignored as requested)
PLATE_CLASS = 1
VIOLATION_CLASSES = [2, 3, 5, 7]  # mobile_usage, pillion_not_wearing_helmet, rider_not_wearing_helmet, triple_riding
WEARING_CLASSES = [4, 6]  # pillion_wearing_helmet, rider_wearing_helmet
VEHICLE_CLASS = 8  # vehicle (used for context only)


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
                
                # Keep the one with significantly higher confidence
                # BIAS: Favor "wearing" detection (compliance) to reduce false positives
                # If wearing confidence is within 0.2 of violation confidence, assume wearing
                if v_conf > (w_conf + 0.2):
                    # Keep violation ONLY if it is much stronger (> 0.2 higher)
                    # Mark wearing for removal
                    wearing_to_remove.add(id(wearing))
                    filtered_violations.append(violation)
                else:
                    # Keep wearing (safer bet), skip violation
                    # This filters false positive violations that have a corresponding wearing detection
                    pass
                break
        
        # If no conflict found, keep the violation
        if not conflict_found:
            filtered_violations.append(violation)
    
    # Remove conflicting wearing detections
    filtered_wearing = [w for w in wearing_detections if id(w) not in wearing_to_remove]
    
    return filtered_violations, filtered_wearing


def deduplicate_violations(violations, iou_threshold=0.2, containment_threshold=0.6, proximity_threshold=100):
    """
    Remove duplicate violations using Non-Maximum Suppression with containment check.
    Prevents the same person from being reported multiple times for the same violation.
    
    IMPROVED: Now handles cross-type duplicates (e.g., same person detected as both
    rider_not_wearing_helmet AND pillion_not_wearing_helmet).
    
    Handles three cases:
    1. Overlapping boxes (IoU > threshold)
    2. Contained boxes (smaller box inside larger box)
    3. Adjacent boxes (close proximity + similar size)
    
    Args:
        violations: List of violation dicts with 'type', 'confidence', 'bbox'
        iou_threshold: IoU threshold for considering violations as duplicates (default: 0.2, lowered from 0.3)
        containment_threshold: Containment ratio for smaller box in larger (default: 0.6, lowered from 0.7)
        proximity_threshold: Max distance between box centers to consider as duplicates (default: 100 pixels)
    
    Returns:
        Deduplicated list of violations
    """
    if not violations:
        return []
    
    def box_center(box):
        """Calculate center point of bounding box."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def box_area(box):
        """Calculate area of bounding box."""
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def euclidean_distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
    
    def is_duplicate(box1, box2):
        """Check if two boxes are duplicates using IoU, containment, OR proximity."""
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
        area1 = box_area(box1)
        area2 = box_area(box2)
        
        # Check if smaller box is mostly contained in larger
        smaller_area = min(area1, area2)
        if smaller_area > 0:
            containment_ratio = intersection / smaller_area
            if containment_ratio >= containment_threshold:
                return True
        
        # NEW: Check proximity for adjacent boxes (same person detected twice)
        # If boxes are very close AND similar size, likely same person
        center1 = box_center(box1)
        center2 = box_center(box2)
        distance = euclidean_distance(center1, center2)
        
        if distance <= proximity_threshold:
            # Check if boxes are similar size (within 50% of each other)
            size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
            if size_ratio >= 0.5:  # Boxes are similar size
                return True
        
        return False
    
    # IMPROVED: Deduplicate across ALL violation types, not just within each type
    # This prevents the same person from being detected as both "rider" and "pillion"
    
    # Sort ALL violations by confidence (highest first)
    violations_sorted = sorted(violations, key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    while violations_sorted:
        # Keep the highest confidence detection
        best = violations_sorted.pop(0)
        keep.append(best)
        
        # Remove overlapping or contained detections (duplicates) regardless of type
        remaining = []
        for v in violations_sorted:
            if not is_duplicate(best["bbox"], v["bbox"]):
                # Not a duplicate, keep it (different person/location)
                remaining.append(v)
            # else: duplicate (overlapping or contained), discard
        
        violations_sorted = remaining
    
    return keep


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
    
    # BALANCED MODE: Moderate confidence thresholds to filter weak false positives
    # Increasing this removes low-confidence detections (noise)
    if conf_threshold is None:
        if lighting_condition == 'day':
            conf_threshold = 0.35  # Set to 0.35 to reduce false positives
        else:
            conf_threshold = 0.30  # Set to 0.30 for night conditions
    
    # Adaptive image size
    if lighting_condition == 'day':
        imgsz = 960  # Larger size for better detection
    else:
        imgsz = 640  # Standard size for night
    
    # CRITICAL FIX: Disable augmentation completely
    augment = False
    
    # BALANCED: Deduplication thresholds to remove obvious duplicates
    # But not too aggressive to avoid filtering real violations
    iou_thresh = 0.35  # Higher = Stricter deduplication (fewer boxes)
    containment_thresh = 0.70  # Higher = Stricter containment
    
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
        
        # Skip ignored classes (Class 0)
        if cls in IGNORE_CLASSES:
            continue
        
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
        elif cls == VEHICLE_CLASS:
            all_vehicles.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": vconf
            })

    # -----------------------------
    # -----------------------------
    # Step 3: FALSE POSITIVE FILTERING (Simplified)
    # -----------------------------
    # We now rely on 'resolve_conflicting_detections' (Step 3.5) which is smarter
    # and handles specific type conflicts (Rider vs Rider) rather than generic overlap.
    
    # (Generic overlap filter removed to avoid accidentally filtering valid detections)
    
    if debug:
        print(f"� Raw violations: {len(all_violations)}")
        print(f"📊 Wearing detections: {len(all_wearing_detections)}")
    
    if debug:
        print(f"📊 Filtered violations (after FP check): {len(all_violations)}")
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
    # Step 3.6: Verify violations with Secondary Classifier (ResNet/MobileNet)
    # -----------------------------
    # Run each candidate violation crop through the verifier model
    # If the verifier says "safe" (e.g. helmet present), discard it.
    
    verified_violations = []
    
    for violation in all_violations:
        # Check if this type needs verification (e.g. helmet violations)
        v_type = violation['type']
        if 'not_wearing' in v_type or 'triple' in v_type:
            # Crop the violation region
            vx1, vy1, vx2, vy2 = _clip_box(violation['bbox'], w, h)
            # Add margin for context
            margin = 10
            vx1 = max(0, vx1 - margin)
            vy1 = max(0, vy1 - margin)
            vx2 = min(w, vx2 + margin)
            vy2 = min(h, vy2 + margin)
            
            violation_crop = image_cv[vy1:vy2, vx1:vx2]
            
            # Run verification
            result = verifier.verify(violation_crop)
            
            if result == 'safe':
                if debug:
                    print(f"🛡️ Verifier caught false positive: {v_type} -> Safe")
                continue # Skip adding this violation
            elif result == 'violation':
                if debug:
                    print(f"✅ Verifier confirmed: {v_type}")
        
        # Keep violation (either confirmed or not applicable for verification)
        verified_violations.append(violation)
            
    all_violations = verified_violations

    if debug:
        print(f"📊 Violations after verification: {len(all_violations)}")

    # -----------------------------
    # Step 4: Deduplicate remaining violations
    # -----------------------------
    # CRITICAL: Separate triple_riding to avoid suppressing individual violations (rider/pillion)
    # The triple_riding box often contains individual rider boxes, so aggressive deduplication
    # would delete the individual violations if triple_riding has higher confidence.
    triple_violations = [v for v in all_violations if v['type'] == 'triple_riding']
    other_violations = [v for v in all_violations if v['type'] != 'triple_riding']
    
    # Deduplicate separately
    other_violations = deduplicate_violations(other_violations, iou_threshold=iou_thresh, containment_threshold=containment_thresh)
    triple_violations = deduplicate_violations(triple_violations, iou_threshold=iou_thresh, containment_threshold=containment_thresh)
    
    # Combine back
    all_violations = other_violations + triple_violations

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
    # Step 6.5: Filter Triple Riding False Positives (User Rule)
    # -----------------------------
    # "Only detect triple riding if there are greater than or equal to two pillion riders"
    for group in vehicle_groups:
        # Count pillions (both violating and compliant)
        pillion_count = 0
        
        # Check violations
        for v in group["violations"]:
            if "pillion" in v["type"].lower():
                pillion_count += 1
        
        # Check compliance
        for c in group["compliance"]:
            if "pillion" in c["type"].lower():
                pillion_count += 1
        
        # Filter triple_riding if pillion_count < 2
        # We assume standard bike has 1 rider. So 2 pillions = 3 people total.
        new_violations = []
        triple_removed = False
        
        for v in group["violations"]:
            if "triple_riding" in v["type"].lower():
                if pillion_count >= 2:
                    new_violations.append(v)
                else:
                    triple_removed = True
            else:
                new_violations.append(v)
        
        if triple_removed and debug:
            print(f"   🧹 Filtered Triple Riding FP (Group has {pillion_count} pillions)")
            
        group["violations"] = new_violations

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
    # Step 10: Build response with violations only (compliance removed)
    # Filter out detections with no violations
    # -----------------------------
    formatted_detections = []
    for det in detections:
        violations_list = [
            {
                "type": v["type"],
                "confidence": v["confidence"],
                "bbox": v["bbox"]
            }
            for v in det.get("violations", [])
        ]
        
        # Only include detections that have at least one violation
        if violations_list:
            formatted_detections.append({
                "number_plate": {
                    "plate_number": det["plate_number"],
                    "plate_bbox": det["bbox"]
                },
                "violations": violations_list
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
    
    # Class names mapping (NEW MODEL)
    CLASS_NAMES = {
        0: '6',  # Ignored
        1: 'Number_plate',
        2: 'mobile_usage',
        3: 'pillion_not_wearing_helmet',
        4: 'pillion_wearing_helmet',
        5: 'rider_not_wearing_helmet',
        6: 'rider_wearing_helmet',
        7: 'triple_riding',
        8: 'vehicle'
    }
    
    # Color mapping for each class (BGR format)
    CLASS_COLORS = {
        0: (128, 128, 128),  # Gray - Ignored class
        1: (255, 255, 0),    # Cyan - Number_plate
        2: (0, 0, 255),      # Red - mobile_usage
        3: (0, 0, 139),      # Dark Red - pillion_not_wearing_helmet
        4: (0, 255, 0),      # Green - pillion_wearing_helmet
        5: (0, 69, 255),     # Orange - rider_not_wearing_helmet
        6: (255, 0, 255),    # Magenta - rider_wearing_helmet
        7: (255, 255, 255),  # White - triple_riding
        8: (255, 255, 255)   # White - vehicle
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
        
        if cls_id == 8:  # vehicle (updated for new model)
            vehicles.append(det)
        elif cls_id not in IGNORE_CLASSES:  # Skip ignored classes
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
