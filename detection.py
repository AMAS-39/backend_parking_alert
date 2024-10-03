# detection.py

import torch
import cv2
import pytesseract
import re
import numpy as np
import easyocr
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = r'C:\Users\Best Center\Contacts\Desktop\backend_parking_alert\best.pt'

# Load the YOLOv5 model globally to avoid reloading for each request
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False, trust_repo=True)

# Initialize EasyOCR reader globally
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if CUDA is available

def extract_plate_components(text):
    """
    Extracts city code, letter, and number from the license plate text.
    
    Args:
        text (str): The cleaned license plate text.
    
    Returns:
        tuple or None: (city_code, letter, number) if extraction is successful, else None.
    """
    logger.info(f"Extracting plate components from text: '{text}'")
    
    # Define a flexible pattern to capture city code, letter, and number
    plate_pattern = r'(?P<city_code>2[1-4])[A-Z](?P<number>\d{1,5})'

    match = re.search(plate_pattern, text)
    if match:
        city_code = match.group('city_code')
        number = match.group('number')

        # To extract the letter, find the first uppercase letter after city_code
        letter_match = re.search(r'[A-Z]', text[match.end('city_code'):])
        if letter_match:
            letter = letter_match.group(0)
        else:
            logger.warning("Letter not found after city code.")
            return None

        logger.info(f"Extracted Components - City Code: {city_code}, Letter: {letter}, Number: {number}")
        return city_code, letter, number
    else:
        logger.warning("Plate pattern did not match.")
        return None

def detect_plate_number(image_path=None, image=None):
    """
    Detects the license plate number from an image.

    Args:
        image_path (str): Path to the image file.
        image (numpy.ndarray): Image array.

    Returns:
        str or None: Detected plate number or None if not found.
    """
    if image is None and image_path is None:
        logger.error("No image provided to detect_plate_number.")
        return None

    # Load image from path or use provided image array
    if image is None:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image from path: {image_path}")
            return None

    # Run detection
    results = model(image)

    # Get the detected bounding boxes
    detections = results.xyxy[0]  # tensor of detections

    if len(detections) == 0:
        logger.info("No detections found in the image.")
        return None
    else:
        # Loop over detections
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Adjust padding
            padding_x = int((x2 - x1) * 0.02)  # 2% padding
            padding_y = int((y2 - y1) * 0.05)  # 5% padding

            x1 = max(x1 - padding_x, 0)
            y1 = max(y1 - padding_y, 0)
            x2 = min(x2 + padding_x, image.shape[1] - 1)
            y2 = min(y2 + padding_y, image.shape[0] - 1)

            # Crop the detected license plate region
            plate_region = image[y1:y2, x1:x2]

            # Preprocess the image
            gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_plate = clahe.apply(gray_plate)

            # Apply sharpening
            kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            gray_plate = cv2.filter2D(gray_plate, -1, kernel)

            # Apply thresholding
            blur = cv2.GaussianBlur(gray_plate, (5, 5), 0)
            _, thresh_plate = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply OCR using EasyOCR
            result_easyocr = reader.readtext(thresh_plate, detail=0)
            text_easyocr = ''.join(result_easyocr)
            logger.info(f"Extracted Text with EasyOCR: {text_easyocr}")

            # Preprocess the text
            text = text_easyocr.upper()
            text = re.sub(r'\s+', '', text)  # Remove all whitespace
            text = re.sub(r'[^A-Z0-9]', '', text)  # Remove any non-alphanumeric characters
            logger.info(f"Cleaned Text: {text}")

            # Extract plate components
            components = extract_plate_components(text)

            if components:
                city_code, letter, number = components

                # Apply corrections to city code if necessary
                if not city_code.isdigit():
                    city_code = city_code.replace('O', '0').replace('I', '1').replace('Z', '2')
                logger.info(f"City Code After Correction: {city_code}")

                # Correct the letter if necessary
                if not letter.isalpha():
                    letter_corrections = {
                        '0': 'O', '6': 'G', '8': 'B', '1': 'I', '5': 'S', '2': 'Z', '7': 'T', '9': 'G'
                    }
                    letter = letter_corrections.get(letter, 'A')
                logger.info(f"Letter After Correction: {letter}")

                # Correct the number if necessary
                number_original = number
                number = number.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
                number = number.replace('G', '6').replace('B', '8').replace('Q', '0')
                number = number.replace('D', '0').replace('A', '4')
                number = number.replace('?', '7').replace('T', '7')
                # Remove any non-digit characters
                number = re.sub(r'[^0-9]', '', number)
                number = number[:5]
                logger.info(f"Number After Correction: {number}")

                # Generate alternative numbers by replacing '2' with '9' and vice versa
                possible_numbers = [number]
                if '2' in number:
                    number_with_9 = number.replace('2', '9')
                    possible_numbers.append(number_with_9)
                if '9' in number:
                    number_with_2 = number.replace('9', '2')
                    possible_numbers.append(number_with_2)
                possible_numbers = list(set(possible_numbers))  # Remove duplicates

                # Try all possible numbers
                plate_scores = []
                for num in possible_numbers:
                    corrected_text = f"{city_code}{letter}{num}"
                    logger.info(f"Trying Number: {num}, Corrected Text: {corrected_text}")
                    pattern = r'^(2[1-4])[A-Z](\d{1,5})$'
                    match = re.match(pattern, corrected_text)
                    if match:
                        plate_number = f"{city_code}{letter}{num}"
                        # Calculate score with weighted corrections
                        corrections_applied = 0
                        for orig_digit, corrected_digit in zip(number_original, num):
                            if orig_digit != corrected_digit:
                                if (orig_digit == '2' and corrected_digit == '9') or (orig_digit == '9' and corrected_digit == '2'):
                                    corrections_applied += 0.1  # Lower penalty for known misreads
                                else:
                                    corrections_applied += 1  # Higher penalty for other corrections
                        plate_scores.append((plate_number, corrections_applied))
                        logger.info(f"Plate Number Found: {plate_number}, Corrections Applied: {corrections_applied}")

                if plate_scores:
                    # Select the plate number with the least corrections
                    final_plate_number = min(plate_scores, key=lambda x: x[1])[0]
                    logger.info(f"Final Plate Number: {final_plate_number}")
                    return final_plate_number
                else:
                    logger.warning("No plate number found matching the format after corrections.")
                    return None
            else:
                logger.warning("Failed to extract plate components.")
                return None
