import pickle

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

MAX_LEN = 200

model = load_model(
    "models/sentiment_lstm.keras"
)

with open(
    "models/word_index.pkl",
    "rb"
) as f:
    word_index = pickle.load(f)

def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:

        encoded.append(
            word_index.get(
                word,
                2
            ) + 3
        )

    return encoded

def predict_sentiment(text):

    encoded = encode_review(
        text
    )

    padded = pad_sequences(
        [encoded],
        maxlen=MAX_LEN
    )

    score = model.predict(
        padded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if score >= 0.5
        else "Negative"
    )

    return sentiment, score