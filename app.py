from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64-encoded image from the request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the data URL prefix
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make a prediction
        prediction = model.predict(processed_frame)
        predicted_class = np.argmax(prediction)
        predicted_letter = class_labels[predicted_class]

        # Return the prediction
        return jsonify({"prediction": predicted_letter})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel requires the app to be wrapped in a WSGI callable
wsgi_app = app.wsgi_app

if __name__ == '__main__':
    app.run(debug=True)