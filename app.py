import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(_name_)

# Load the trained model
model = tf.keras.models.load_model('sign_language_cnn.h5')

# Define class labels (A-Z excluding J and Z)
class_labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

# Function to preprocess the frame
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize pixel values
    normalized = resized / 255.0
    # Reshape for the model
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

# Function to generate video frames with predictions
def generate_frames():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make a prediction
        prediction = model.predict(processed_frame)
        predicted_class = np.argmax(prediction)
        predicted_letter = class_labels[predicted_class]

        # Display the predicted letter on the frame
        cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the webcam
    cap.release()

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

if _name_ == '_main_':
    app.run(debug=True)