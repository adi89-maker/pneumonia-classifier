import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model with error handling
MODEL_PATH = 'pneumonia_classifier_model.h5'
model = None

try:
    model = load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists")

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure the model file exists and is valid'
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the uploaded file
        img_bytes = file.read()
        
        # Open image using PIL from bytes
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to match model input (150x150)
        img = img.resize((150, 150))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction)
        
        # Determine result
        if confidence > 0.5:
            result = 'Pneumonia'
            confidence_percentage = confidence * 100
        else:
            result = 'Normal'
            confidence_percentage = (1 - confidence) * 100
        
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'confidence_percentage': round(confidence_percentage, 2),
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the URL'
    }), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': 'Maximum file size is 16MB'
    }), 413

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Available endpoints:")
    print("  GET  / - Home page")
    print("  POST /predict - Make predictions")
    print(f"  Model status: {'Loaded' if model else 'Not loaded'}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)