from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
import json
import os

def run_lstm_model(df):
    """
    Train and evaluate an LSTM model using Keras.
    """
    print("[INFO] Running LSTM Model...")

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['target'])
    y = to_categorical(y, num_classes=3)

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['clean_text'])

    X = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(X, maxlen=100)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build Model
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, batch_size=64, epochs=3, validation_split=0.1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"[INFO] LSTM Test Accuracy: {accuracy:.2f}")

    # Create directory if not exists
    os.makedirs("models/saved_models", exist_ok=True)

    # Save Model
    model.save("models/saved_models/lstm_model.h5")
    print("[INFO] LSTM model saved to models/saved_models/lstm_model.h5")

    # ✅ Save Tokenizer
    tokenizer_json = tokenizer.to_json()
    with open("models/saved_models/tokenizer.json", "w") as f:
        f.write(tokenizer_json)
    print("[INFO] Tokenizer saved to models/saved_models/tokenizer.json")
