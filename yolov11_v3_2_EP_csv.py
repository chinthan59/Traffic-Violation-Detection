import csv
import requests
import os
import tempfile
import json
from src.detector import process_car_image
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
    Process image and extract violation information using detector_yolov11_v3_EP.
    Returns a list of detection results - one per vehicle detected.
    Each vehicle gets its own row with its specific violations.
    
    Uses car_model_EP.pt with no_passenger class to prevent false positives.
    """
    try:
        # Get API response from detector_yolov11_v3_EP
        result = process_car_image(image_path)
        detections = result.get("detections", [])
        message = result.get("message", "")
        
        # If there's a "no violation" message, return early with single empty result
        if message and "No violations detected" in message:
            return [{
                "detected_plate": "",
                "driver_violation": "",
                "passenger_violation": "",
                "message": "No violation detected - all occupants wearing seatbelts"
            }]
        
        # If no detections at all, return empty result
        if not detections:
            return [{
                "detected_plate": "",
                "driver_violation": "",
                "passenger_violation": "",
                "message": "No detections found"
            }]
        
        # Process EACH detection separately (one per vehicle)
        results = []
        for detection in detections:
            # Extract plate information for THIS vehicle
            detected_plate = ""
            if detection.get("number_plate"):
                plate_info = detection["number_plate"]
                plate_number = plate_info.get("plate_number", "").strip()
                anpr_conf = plate_info.get("anpr_confidence", 0.0)
                
                if plate_number:
                    # Format with confidence if available
                    if anpr_conf > 0:
                        detected_plate = f"{plate_number} ({anpr_conf:.3f})"
                    else:
                        detected_plate = plate_number
            
            # Extract violations for THIS vehicle only
            driver_violations = []
            passenger_violations = []
            violations = detection.get("violations", [])
            
            for violation in violations:
                vtype = violation.get("type", "")
                conf = violation.get("confidence", 0.0)
                
                # Categorize violations
                if "Driver" in vtype or "driver" in vtype:
                    driver_violations.append(f"{vtype} ({conf:.2f})")
                elif "passenger" in vtype or "Passenger" in vtype:
                    passenger_violations.append(f"{vtype} ({conf:.2f})")
            
            # Format violations as strings
            driver_violation_str = "; ".join(driver_violations) if driver_violations else ""
            passenger_violation_str = "; ".join(passenger_violations) if passenger_violations else ""
            
            # Determine message for THIS vehicle
            if driver_violations or passenger_violations:
                message_str = "Violation detected"
            else:
                message_str = "No violation detected - all occupants wearing seatbelts"
            
            # Add result for THIS vehicle
            results.append({
                "detected_plate": detected_plate,
                "driver_violation": driver_violation_str,
                "passenger_violation": passenger_violation_str,
                "message": message_str
            })
        
        return results
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return [{
            "detected_plate": "",
            "driver_violation": "",
            "passenger_violation": "",
            "message": f"Error: {str(e)}"
        }]

def process_csv(input_csv_path, output_csv_path, start_row=1, max_rows=None):
    """
    Process CSV file and create output CSV with violation detection results.
    
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
    print(f"Using detector: detector.py (Main detector with ANPR confidence)")
    print(f"Model: car.pt + car_yolov11.pt (dual model detection)")
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
            if 'registration' in col.lower():
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
                    "driver_violation": "",
                    "passenger_violation": "",
                    "message": "Download error"
                })
                continue
            
            try:
                # Process image and get detections (returns list of vehicles)
                detection_list = process_detections(temp_image_path)
                
                # Create one CSV row per detected vehicle
                for detection_info in detection_list:
                    # Combine with original data
                    result_row = {
                        "number_plate": registration_number,
                        "image_link": image_url,
                        "detected_number_plate": detection_info["detected_plate"],
                        "driver_violation": detection_info["driver_violation"],
                        "passenger_violation": detection_info["passenger_violation"],
                        "message": detection_info["message"]
                    }
                    
                    results.append(result_row)
                    
                    # Print summary for THIS vehicle
                    if detection_info["detected_plate"]:
                        print(f"  ✓ Detected Plate: {detection_info['detected_plate']}")
                    else:
                        print(f"  ✗ No plate detected")
                    
                    if detection_info["driver_violation"]:
                        print(f"  ⚠ Driver Violation: {detection_info['driver_violation']}")
                    
                    if detection_info["passenger_violation"]:
                        print(f"  ⚠ Passenger Violation: {detection_info['passenger_violation']}")
                    
                    if not detection_info["driver_violation"] and not detection_info["passenger_violation"]:
                        print(f"  ✓ {detection_info['message']}")
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"  ✗ Error: {e}")
                error_row = {
                    "number_plate": registration_number,
                    "image_link": image_url,
                    "detected_number_plate": "",
                    "driver_violation": "",
                    "passenger_violation": "",
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
            "driver_violation",
            "passenger_violation",
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
    
    parser = argparse.ArgumentParser(
        description='Process images from CSV using detector_yolov11_v3_EP.py (Enhanced Precision with no_passenger class)'
    )
    parser.add_argument('--input', '-i', default='seatbelt_not_wearing_day.csv',
                       help='Input CSV file path (default: seatbelt_not_wearing_day.csv)')
    parser.add_argument('--output', '-o', default='yolov11_v3_EP_detection_results.csv',
                       help='Output CSV file path (default: yolov11_v3_EP_detection_results.csv)')
    parser.add_argument('--start', '-s', type=int, default=1,
                       help='Start from row number (default: 1)')
    parser.add_argument('--max', '-m', type=int, default=None,
                       help='Maximum number of rows to process (default: all)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLOv11 CSV Processor with ANPR Confidence")
    print("Main Detector (car.pt + car_yolov11.pt)")
    print("Includes ANPR confidence in detected number plates")
    print("=" * 80)
    print()
    
    start_time = time.time()
    process_csv(args.input, args.output, args.start, args.max)
    end_time = time.time()
    
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
