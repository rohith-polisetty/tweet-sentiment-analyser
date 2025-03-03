from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load the trained model and vectorizer
try:
    with open('rnnvectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF Vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

try:
    with open('rnnmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    print("Serving index.html")
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweet = data['tweet']
        print(f"Received tweet: {tweet}")
        
        # Preprocess the tweet using TF-IDF
        tfidf_features = tfidf_vectorizer.transform([tweet]).toarray()
        print(f"TF-IDF features shape: {tfidf_features.shape}")
        
        # Reshape for RNN if needed (assuming model expects 3D input)
        # Adjust this based on your model's expected input shape
        tfidf_features = tfidf_features.reshape(1, -1, 1)
        print(f"Reshaped input shape: {tfidf_features.shape}")
        
        # Make prediction
        prediction = model.predict(tfidf_features)[0][0]
        print(f"Raw prediction: {prediction}")
        
        # Interpret result
        sentiment = 'Positive' if prediction > 0.5 else 'Negative'
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'success': True
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    app.run(debug=True)