import pickle
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

MAX_WORDS = 20000
MAX_LEN = 150

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=MAX_WORDS
)

x_train = pad_sequences(
    x_train,
    maxlen=MAX_LEN
)

x_test = pad_sequences(
    x_test,
    maxlen=MAX_LEN
)
model = Sequential()

model.add(
    Input(shape=(MAX_LEN,))
)

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=64
    )
)

model.add(
    LSTM(
        64,
        return_sequences=True
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    LSTM(
        32
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

os.makedirs("models", exist_ok=True)

model.save(
    "models/sentiment_lstm.keras"
)


word_index = tf.keras.datasets.imdb.get_word_index()

with open(
    "models/word_index.pkl",
    "wb"
) as f:
    pickle.dump(
        word_index,
        f
    )

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print(f"Accuracy: {accuracy:.4f}")