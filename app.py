from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture using OpenCV
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(image):
    # Decode the base64 image
    image_data = base64.b64decode(image.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the BGR image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        # Send hand landmarks to the frontend
        emit('hand_landmarks', results.multi_hand_landmarks)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)
