import torch
import cv2
import re
import numpy as np
import easyocr
import logging
from pathlib import Path

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path('best.pt')

# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load the YOLOv5 model using torch.hub from the Ultralytics repository, and move it to the correct device
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH), force_reload=True)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    logger.info(f"Model loaded successfully on {device}.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# Initialize EasyOCR reader, use GPU if available
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

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

        # Extract the letter after the city_code
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

def detect_plate_number(image_path=None):
    """
    Detects the license plate number from an image.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str or None: Detected plate number or None if not found.
    """
    if image_path is None:
        logger.error("No image provided to detect_plate_number.")
        return None

    # Load image from path
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image from path: {image_path}")
        return None

    # Run YOLOv5 inference
    results = model(image)
    detections = results.xyxy[0]

    if len(detections) == 0:
        logger.info("No license plates detected in the image.")
        return None

    # Extract detected plates
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop the detected plate region from the image
        plate_image = image[y1:y2, x1:x2]

        # Convert to grayscale and apply adaptive thresholding
        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_plate = clahe.apply(gray_plate)

        # Apply sharpening to highlight characters
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_plate = cv2.filter2D(enhanced_plate, -1, sharpen_kernel)

        # Apply Gaussian Blur for noise reduction
        blurred_plate = cv2.GaussianBlur(sharpened_plate, (3, 3), 0)

        # Use Otsu's thresholding for better binarization
        _, binary_plate = cv2.threshold(blurred_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply OCR using EasyOCR
        ocr_results = reader.readtext(binary_plate, detail=0)
        text_easyocr = ''.join(ocr_results)
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
            return f"{city_code}{letter}{number}"

    return None
