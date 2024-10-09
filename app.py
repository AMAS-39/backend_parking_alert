import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import firebase_admin
from firebase_admin import credentials, firestore, messaging
from detection import detect_plate_number
import logging
import re
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Firebase
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH', 'credentials.json')
if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
    logger.error(f"Firebase credentials file not found at {FIREBASE_CREDENTIALS_PATH}")
    raise FileNotFoundError(f"Firebase credentials file not found at {FIREBASE_CREDENTIALS_PATH}")

cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload directory at {UPLOAD_FOLDER}")

# Set maximum file size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_email(owner_email, plate_number):
    """
    Send an email to the car owner with the plate number.
    """
    try:
        sender_email = "parking.alert.krd@gmail.com"
        sender_password = "wund awhp flbr mpqv"  # Use Gmail app-specific password
        subject = "Parking Alert: Please move your car"
        body = f"Dear car owner,\n\nYour car with plate number {plate_number} is blocking another vehicle. Kindly move it.\n\nBest regards,\nParking Alert System"
        
        # Set up the MIME message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = owner_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Set up the server connection
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Send the email
        server.sendmail(sender_email, owner_email, msg.as_string())
        server.quit()
        
        logger.info(f"Email sent to {owner_email} regarding plate number {plate_number}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {owner_email}: {e}")
        return False

def send_notification(token, title, body):
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            token=token
        )
        response = messaging.send(message)
        logger.info(f"Successfully sent message: {response}")
        return True
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return False

def normalize_plate_number(plate_number):
    plate_number = plate_number.upper()
    plate_number = plate_number.replace(" ", "")
    plate_number = re.sub(r'[^A-Z0-9]', '', plate_number)
    return plate_number

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            logger.warning("No image part in the request.")
            return jsonify({'success': False, 'message': 'No image part in the request.'}), 400

        file = request.files['image']

        if file.filename == '':
            logger.warning("No selected file.")
            return jsonify({'success': False, 'message': 'No selected file.'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_filename = f"uploaded_{len(os.listdir(UPLOAD_FOLDER)) + 1}_{filename}"
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            file.save(image_path)
            logger.info(f"Image saved to {image_path}")

            plate_number = detect_plate_number(image_path=image_path)

            if not plate_number:
                logger.info("No license plate detected in the image.")
                return jsonify({'success': False, 'message': 'No license plate detected.'}), 200

            logger.info(f"Detected Plate Number: {plate_number}")
            plate_number = normalize_plate_number(plate_number)

            logger.info(f"Searching Firestore for plateNumber: '{plate_number}'")
            try:
                users_ref = db.collection('users')
                query = users_ref.where('car.plateNumber', '==', plate_number)
                car_docs = query.stream()
                matching_cars = list(car_docs)
                logger.info(f"Number of car documents found with plateNumber '{plate_number}': {len(matching_cars)}")

                for car_doc in matching_cars:
                    logger.info(f"Matching Car Data: {car_doc.to_dict()}")

            except Exception as e:
                logger.error(f"Error querying Firestore: {e}")
                return jsonify({'success': False, 'message': 'Error querying database.'}), 500

            if not matching_cars:
                logger.info(f"Plate number {plate_number} not found in database.")
                return jsonify({
                    'success': False,
                    'plate_number': plate_number,
                    'message': f"Plate number {plate_number} not found in database.",
                    'image_url': image_path
                }), 200

            notifications_sent = []
            for car_doc in matching_cars:
                user_ref = car_doc.reference
                user_doc = user_ref.get()

                if not user_doc.exists:
                    logger.warning(f"User document not found for car document ID: {car_doc.id}")
                    continue

                user_data = user_doc.to_dict()
                fcm_token = user_data.get('fcm_token')
                owner_name = user_data.get('name')
                owner_email = user_data.get('email')

                if fcm_token:
                    notification_sent = send_notification(
                        fcm_token,
                        'Parking Alert',
                        f'Someone has taken a picture of your car (Plate: {plate_number}).'
                    )
                    if notification_sent:
                        notifications_sent.append(f"Notification sent to {owner_name}.")
                    else:
                        logger.error(f"Failed to send in-app notification to {owner_name}.")

                email_sent = send_email(owner_email, plate_number)
                if email_sent:
                    logger.info(f"Email notification sent to {owner_email}.")
                else:
                    logger.error(f"Failed to send email to {owner_email}.")

            if notifications_sent:
                return jsonify({
                    'success': True,
                    'plate_number': plate_number,
                    'message': " ".join(notifications_sent),
                    'image_url': image_path
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'plate_number': plate_number,
                    'message': "Failed to send notifications.",
                    'image_url': image_path
                }), 500

        else:
            logger.warning("Invalid file type uploaded.")
            return jsonify({'success': False, 'message': 'Invalid file type.'}), 400

    except Exception as e:
        logger.exception("An error occurred while processing the image.")
        return jsonify({'success': False, 'message': 'Internal server error.'}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
