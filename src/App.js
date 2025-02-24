import React, { useRef, useState } from 'react';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const [prediction, setPrediction] = useState('');

  // Start the webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (err) {
      console.error('Error accessing webcam:', err);
    }
  };

  // Capture a frame and send it to the Flask backend
  const captureFrame = async () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas image to base64
    const frame = canvas.toDataURL('image/jpeg');

    // Send frame to Flask backend
    try {
      const response = await fetch('https://your-flask-app.vercel.app/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frame }),
      });

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error('Error sending frame to backend:', err);
    }
  };

  return (
    <div className="App">
      <h1>Sign Language Recognition</h1>
      <button onClick={startWebcam}>Start Webcam</button>
      <video ref={videoRef} autoPlay playsInline muted />
      <button onClick={captureFrame}>Capture Frame</button>
      {prediction && <p>Predicted: {prediction}</p>}
    </div>
  );
}

export default App;