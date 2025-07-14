import pandas as pd
from utils.preprocessing import clean_text
from models.logistic_model import run_logistic_regression
from models.lstm_model import run_lstm_model

def load_data(filepath):
    """
    Load the tweet dataset and return the processed DataFrame
    """
    df = pd.read_csv(filepath, encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = df[['text', 'target']]
    df['target'] = df['target'].replace({0: 'negative', 2: 'neutral', 4: 'positive'})
    return df

def preprocess_data(df):
    """
    Clean the tweet texts using NLTK preprocessing
    """
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def main():
    # Step 1: Load Data
    print("[INFO] Loading data...")
    df = load_data('data/tweets.csv')
    print(f"[INFO] Loaded {len(df)} tweets")

    # Step 2: Preprocess Data
    print("[INFO] Cleaning tweets...")
    df = preprocess_data(df)
    print("[INFO] Sample cleaned tweet:", df['clean_text'].iloc[0])

    # Step 3: Save Cleaned Data
    df.to_csv('data/cleaned_tweets.csv', index=False)
    print("[INFO] Cleaned data saved to data/cleaned_tweets.csv")

    # Step 4: Run Logistic Regression Model
    print("\n[INFO] ----- Running Logistic Regression Model -----")
    run_logistic_regression(df)

    # Step 5: Run LSTM Model
    print("\n[INFO] ----- Running LSTM Model -----")
    run_lstm_model(df)

    print("\n[INFO] ✅ All models executed successfully.")

if __name__ == '__main__':
    main()
