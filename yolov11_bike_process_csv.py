"""
Bike YOLOv11 CSV Processor

Processes bike images from CSV file and logs violations.
Detects:
- Violations: rider/pillion not wearing helmet, mobile usage, triple riding

Based on yolov11_v3_2_EP_csv_new.py (car detector pattern).
"""

import csv
import requests
import os
import tempfile
import json
from src.detector2_bike_new import process_bike_image
from datetime import datetime
import time

def download_image(url, timeout=30):
    """Download image from URL and save to temporary file."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def process_detections(image_path):
    """
    Process bike image and extract violation information.
    Returns a list of detection results - one per bike detected.
    Each bike gets its own row with its specific violations.
    """
    try:
        # Get API response from detector_yolov11_bike
        result = process_bike_image(image_path)
        detections = result.get("detections", [])
        message = result.get("message", "")
        
        # If there's a "no violation" message, return early with single empty result
        if message and "No violations detected" in message:
            return [{
                "detected_plate": "",
                "rider_violation": "",
                "pillion_violation": "",
                "mobile_usage": "",
                "triple_riding": "",
                "message": "No violations detected - all riders wearing helmets properly"
            }]
        
        # If no detections at all, return empty result
        if not detections:
            return [{
                "detected_plate": "",
                "rider_violation": "",
                "pillion_violation": "",
                "mobile_usage": "",
                "triple_riding": "",
                "message": "No detections found"
            }]
        
        # Process EACH detection separately (one per bike)
        results = []
        for detection in detections:
            # Extract plate information for THIS bike
            detected_plate = ""
            if detection.get("number_plate"):
                plate_info = detection["number_plate"]
                plate_number = plate_info.get("plate_number", "").strip()
                if plate_number:
                    detected_plate = plate_number
            
            # Extract violations for THIS bike
            rider_violations = []
            pillion_violations = []
            mobile_usage_list = []
            triple_riding_list = []
            
            violations = detection.get("violations", [])
            for violation in violations:
                vtype = violation.get("type", "")
                conf = violation.get("confidence", 0.0)
                
                # Categorize violations
                if "rider_not_wearing_helmet" in vtype:
                    rider_violations.append(f"{vtype} ({conf:.2f})")
                elif "pillion_not_wearing_helmet" in vtype:
                    pillion_violations.append(f"{vtype} ({conf:.2f})")
                elif "mobile_usage" in vtype:
                    mobile_usage_list.append(f"{vtype} ({conf:.2f})")
                elif "triple_riding" in vtype:
                    triple_riding_list.append(f"{vtype} ({conf:.2f})")
            
            # Format violations as strings
            rider_violation_str = "; ".join(rider_violations) if rider_violations else ""
            pillion_violation_str = "; ".join(pillion_violations) if pillion_violations else ""
            mobile_usage_str = "; ".join(mobile_usage_list) if mobile_usage_list else ""
            triple_riding_str = "; ".join(triple_riding_list) if triple_riding_list else ""
            
            # Determine message for THIS bike
            if rider_violations or pillion_violations or mobile_usage_list or triple_riding_list:
                message_str = "Violation detected"
            else:
                message_str = "No violations detected - all riders wearing helmets properly"
            
            # Add result for THIS bike
            results.append({
                "detected_plate": detected_plate,
                "rider_violation": rider_violation_str,
                "pillion_violation": pillion_violation_str,
                "mobile_usage": mobile_usage_str,
                "triple_riding": triple_riding_str,
                "message": message_str
            })
        
        return results
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return [{
            "detected_plate": "",
            "rider_violation": "",
            "pillion_violation": "",
            "mobile_usage": "",
            "triple_riding": "",
            "message": f"Error: {str(e)}"
        }]

def process_csv(input_csv_path, output_csv_path, start_row=1, max_rows=None):
    """
    Process CSV file and create output CSV with bike violation AND compliance detection results.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file
        start_row: Row number to start from (1-indexed, excluding header)
        max_rows: Maximum number of rows to process (None for all)
    """
    results = []
    processed_count = 0
    error_count = 0
    
    print(f"Reading CSV file: {input_csv_path}")
    print(f"Using detector: detector2_bike_new.py (YOLOv11 bike detector)")
    print("=" * 80)
    
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Get column names
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Error: CSV file has no headers")
            return
        
        # Find registration_no and image link columns
        reg_col = None
        img_col = None
        
        for col in fieldnames:
            if 'registration' in col.lower() or 'number' in col.lower() or 'plate' in col.lower():
                reg_col = col
            elif 'link' in col.lower() or 'url' in col.lower() or 'image' in col.lower():
                img_col = col
        
        if not reg_col or not img_col:
            print(f"Error: Could not find required columns. Found: {fieldnames}")
            print(f"Registration column: {reg_col}, Image column: {img_col}")
            return
        
        print(f"Using columns - Registration: '{reg_col}', Image URL: '{img_col}'")
        print(f"Starting from row {start_row}, processing {max_rows or 'all'} rows...")
        print("-" * 80)
        
        row_num = 0
        for row in reader:
            row_num += 1
            
            # Skip rows before start_row
            if row_num < start_row:
                continue
            
            # Stop if max_rows reached
            if max_rows and processed_count >= max_rows:
                break
            
            registration_number = row.get(reg_col, "").strip().strip('"')
            image_url = row.get(img_col, "").strip().strip('"')
            
            if not image_url:
                print(f"Row {row_num}: Skipping - no image URL")
                continue
            
            print(f"Row {row_num}: Processing {registration_number} - {image_url[:60]}...")
            
            # Download image
            temp_image_path = download_image(image_url)
            if not temp_image_path:
                error_count += 1
                results.append({
                    "number_plate": registration_number,
                    "image_link": image_url,
                    "detected_number_plate": "",
                    "rider_violation": "",
                    "pillion_violation": "",
                    "mobile_usage": "",
                    "triple_riding": "",
                    "message": "Download error"
                })
                continue
            
            try:
                # Process image and get detections (returns list of bikes)
                detection_list = process_detections(temp_image_path)
                
                # Create one CSV row per detected bike
                for detection_info in detection_list:
                    # Combine with original data
                    result_row = {
                        "number_plate": registration_number,
                        "image_link": image_url,
                        "detected_number_plate": detection_info["detected_plate"],
                        "rider_violation": detection_info["rider_violation"],
                        "pillion_violation": detection_info["pillion_violation"],
                        "mobile_usage": detection_info["mobile_usage"],
                        "triple_riding": detection_info["triple_riding"],
                        "message": detection_info["message"]
                    }
                    
                    results.append(result_row)
                    
                    # Print summary for THIS bike
                    if detection_info["detected_plate"]:
                        print(f"  ✓ Detected Plate: {detection_info['detected_plate']}")
                    else:
                        print(f"  ✗ No plate detected")
                    
                    # Print violations
                    if detection_info["rider_violation"]:
                        print(f"  ⚠ Rider Violation: {detection_info['rider_violation']}")
                    if detection_info["pillion_violation"]:
                        print(f"  ⚠ Pillion Violation: {detection_info['pillion_violation']}")
                    if detection_info["mobile_usage"]:
                        print(f"  ⚠ Mobile Usage: {detection_info['mobile_usage']}")
                    if detection_info["triple_riding"]:
                        print(f"  ⚠ Triple Riding: {detection_info['triple_riding']}")
                    
                    if not any([detection_info["rider_violation"], detection_info["pillion_violation"], 
                               detection_info["mobile_usage"], detection_info["triple_riding"]]):
                        print(f"  ✓ {detection_info['message']}")
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"  ✗ Error: {e}")
                error_row = {
                    "number_plate": registration_number,
                    "image_link": image_url,
                    "detected_number_plate": "",
                    "rider_violation": "",
                    "pillion_violation": "",
                    "mobile_usage": "",
                    "triple_riding": "",
                    "message": f"Processing error: {str(e)}"
                }
                results.append(error_row)
            finally:
                # Clean up temporary file
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
            
            # Progress update every 10 rows
            if processed_count % 10 == 0:
                print(f"Progress: {processed_count} processed, {error_count} errors")
    
    # Write results to output CSV
    print("\n" + "=" * 80)
    print(f"Writing results to: {output_csv_path}")
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "number_plate",
            "image_link",
            "detected_number_plate",
            "rider_violation",
            "pillion_violation",
            "mobile_usage",
            "triple_riding",
            "message"
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nCompleted!")
    print(f"Total processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_csv_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process bike images from CSV using detector2_bike_new.py')
    parser.add_argument('--input', '-i', default='bike_images.csv',
                       help='Input CSV file path (default: bike_images.csv)')
    parser.add_argument('--output', '-o', default='bike_yolov11_detection_results.csv',
                       help='Output CSV file path (default: bike_yolov11_detection_results.csv)')
    parser.add_argument('--start', '-s', type=int, default=1,
                       help='Start from row number (default: 1)')
    parser.add_argument('--max', '-m', type=int, default=None,
                       help='Maximum number of rows to process (default: all)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    process_csv(args.input, args.output, args.start, args.max)
    end_time = time.time()
    
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
