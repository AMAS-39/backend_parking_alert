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
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # Set GPU to True if available

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

    # Convert the image to a tensor and move it to the correct device (CPU/GPU)
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # Run detection
    results = model(img_tensor)

    # Extract bounding boxes, confidence scores, and classes from the results
    detections = results.pred[0]  # This returns a tensor with detections: [x1, y1, x2, y2, confidence, class]

    if len(detections) == 0:
        logger.info("No detections found in the image.")
        return None
    else:
        # Loop over detections
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det[:6].cpu().numpy()  # move to CPU and extract values
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
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_plate = clahe.apply(gray_plate)

            # Apply sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
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

                return f"{city_code}{letter}{number}"

    return None
