import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from pydub import AudioSegment
import speech_recognition as sr
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the dataset when the application starts
reviews_df = pd.read_csv('reviews.csv')

# Define directory for audio files
TEMP_AUDIO = "temp_audio.wav"

# Load RoBERTa sentiment analysis model from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Print column names for debugging
logging.info("Columns in the DataFrame: %s", reviews_df.columns)

# Function to fetch reviews and rating from the dataset
def fetch_reviews_and_rating(product_name):
    logging.info(f"Fetching reviews and rating for product: {product_name}")
    
    # Match product name ignoring case and leading/trailing spaces
    product_data = reviews_df[reviews_df['product'].str.lower().str.strip() == product_name.lower().strip()]
    
    if product_data.empty:
        logging.error(f"No reviews found for product: {product_name}")
        return ["No reviews found."], [], None  # Return no reviews, empty audio_files list, and no rating

    reviews = product_data['review'].tolist()
    audio_files = product_data['audio_files'].tolist()

    # Handle missing or empty audio files
    if not audio_files or pd.isna(audio_files[0]) or audio_files[0] == '':
        logging.warning(f"No audio files available for {product_name}")
        audio_files = []  # Assign empty list if no audio files found

    # Convert rating to int, if available; otherwise, set it as None
    try:
        rating = int(product_data['rating'].iloc[0]) if not pd.isna(product_data['rating'].iloc[0]) else None
    except (ValueError, TypeError):
        rating = None

    return reviews, audio_files, rating

# Function to transcribe audio files
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from the speech recognition service; {e}"

# Sentiment analysis function using RoBERTa
def analyze_sentiment(reviews):
    if not reviews:
        return "No reviews found."

    sentiments = sentiment_pipeline(reviews)
    logging.info(f"Sentiments: {sentiments}")  # Debugging log

    # Aggregate the sentiments
    positive = sum(1 for sentiment in sentiments if sentiment['label'] == 'LABEL_2')  # Positive
    negative = sum(1 for sentiment in sentiments if sentiment['label'] == 'LABEL_0')  # Negative
    neutral = sum(1 for sentiment in sentiments if sentiment['label'] == 'LABEL_1')   # Neutral

    # Determine overall sentiment
    if positive > negative and positive > neutral:
        return "good"
    elif negative > positive and negative > neutral:
        return "bad"
    else:
        return "average"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    product_name = data.get('product_name')
    reviews, audio_files, rating = fetch_reviews_and_rating(product_name)

    # Add transcriptions from audio files (skip if audio file path is invalid or missing)
    audio_reviews = []
    for audio_file in audio_files:
        if isinstance(audio_file, str) and os.path.isfile(audio_file):
            logging.info(f"Transcribing file: {audio_file}")

            if audio_file.endswith(".mp3"):
                sound = AudioSegment.from_mp3(audio_file)
                audio_file = TEMP_AUDIO
                sound.export(audio_file, format="wav")

            transcription = transcribe_audio(audio_file)
            if transcription not in ["Could not understand the audio.", "Could not request results from the speech recognition service; {e}"]:
                audio_reviews.append(transcription)
        else:
            logging.info(f"No valid audio file for {product_name}, skipping audio processing.")

    # Combine text reviews with transcribed audio (if any)
    reviews.extend(audio_reviews)

    if not reviews:
        return jsonify({'sentiment_message': f"No reviews found for {product_name}.", 'rating': rating})

    # Perform sentiment analysis on the reviews (text + audio if available)
    sentiment = analyze_sentiment(reviews)

    # Convert rating to Python int for JSON serialization, handle missing rating
    rating = int(rating) if rating is not None else "No rating available"

    # Pass the product name, sentiment result, and rating
    sentiment_message = f"This product {product_name} is {sentiment}."
    return jsonify({'sentiment_message': sentiment_message, 'rating': rating})

if __name__ == '__main__':
    app.run(debug=True)
