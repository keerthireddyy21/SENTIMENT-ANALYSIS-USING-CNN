# Sentiment Analysis using CNN - Keerthi

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
def build_and_train(vocab_size=10000, max_len=300, epochs=5, batch_size=128):
    # Load IMDB dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    # CNN model
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)
    return model
if __name__ == "__main__":
    build_and_train()