from flask import Flask, request, jsonify
import numpy as np
import pickle
import joblib

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ✅ Load the TF-IDF Vectorizer
import joblib

try:
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ TF-IDF Vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading TF-IDF vectorizer: {e}")


# ✅ Load the Ensemble Model (XGBoost + Logistic Regression)
try:
    with open('ensemble_sentiment_model.pkl', 'rb') as f:
        model = joblib.load(f)
    print("✅ Ensemble Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevents crashes if loading fails

@app.route('/')
def home():
    print("Serving index.html")
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tfidf_vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded', 'success': False})

    try:
        data = request.get_json()
        tweet = data.get('tweet', '').strip()

        if not tweet:
            return jsonify({'error': 'Empty tweet provided', 'success': False})

        print(f"🔹 Received tweet: {tweet}")

        # ✅ Convert tweet to TF-IDF features
        tfidf_features = tfidf_vectorizer.transform([tweet])
        print(f"🔹 TF-IDF features shape: {tfidf_features.shape}")

        # ✅ Make prediction using ensemble model
        prediction_proba = model.predict_proba(tfidf_features)[0]
        predicted_class = np.argmax(prediction_proba)  # 0 for Negative, 1 for Positive
        confidence = round(float(np.max(prediction_proba)) * 100, 2)  # Confidence %

        # ✅ Interpret sentiment
        sentiment = 'Positive' if predicted_class == 1 else 'Negative'

        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'success': True
        })
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    app.run(debug=True)
