import streamlit as st
import numpy as np
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== Load Tokenizer and Model ==========
with open("tokenizer4.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("tweet_sentiment_lstm_model4.h5")

# ========== Configuration ==========
maxlen = 80  # Must match training time padding length

# ========== Label Mapping ==========
label_map = {
    0: "Negative",
    1:"Positive"
}

# ========== Preprocessing Function ==========
def preprocess_tweet(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (but keep the words)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    
    # Remove punctuations and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# ========== Streamlit UI ==========
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Tweet Sentiment Analyzer")
st.markdown("Analyze the sentiment of tweets using a deep learning LSTM model with preprocessing.")

# Input text box
tweet = st.text_area("Enter a Tweet", placeholder="Type or paste a tweet here...", height=150)

# Predict button
if st.button("Analyze Sentiment"):
    if not tweet.strip():
        st.warning("⚠️ Please enter a tweet.")
    else:
        # Preprocess tweet
        cleaned = preprocess_tweet(tweet)
        
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=maxlen, padding='post')
        
        # Predict
        prediction = model.predict(padded)
        pred_class = np.argmax(prediction)

        # Display results
        st.success(f"🧠 **Predicted Sentiment:** {label_map[pred_class]}")
        st.markdown(f"📊 **Confidence:** `{prediction[0][pred_class]*100:.2f}%`")

        st.subheader("🔎 Prediction Breakdown")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{label_map[i]}: {prob*100:.2f}%")
