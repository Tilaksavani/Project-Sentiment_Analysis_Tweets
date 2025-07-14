# 💬 Tweet Sentiment Analyzer

A Machine Learning & Deep Learning project to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments using:

- ✅ Logistic Regression with TF-IDF
- ✅ Deep Learning with LSTM
- ✅ Clean UI with Streamlit

---

## 🧠 Models Used

| Model Type              | Description                                  |
| ----------------------- | -------------------------------------------- |
| **Logistic Regression** | Simple ML model using TF-IDF vectorization   |
| **LSTM**                | Deep Learning model using Keras & embeddings |

---

## 📁 Project Structure

```
├── streamlit_app/
│ └── app.py # Streamlit web app
├── data/
│ └── tweets.csv # Raw dataset
├── models/
│ ├── logistic_model.py # Logistic Regression code
│ ├── lstm_model.py # LSTM model code
│ └── saved_models/ # Trained models and tokenizer
│ ├── logistic_model.pkl
│ ├── tfidf_vectorizer.pkl
│ ├── lstm_model.h5
│ └── tokenizer.json
├── utils/
│ └── preprocessing.py # Text cleaning with NLTK
├── main.py # Run both models from here
├── requirements.txt
└── README.md
```

---

## 🔧 Features

| Feature                | Description                               |
| ---------------------- | ----------------------------------------- |
| **TF-IDF + Logistic**  | Fast traditional ML baseline              |
| **LSTM Model**         | Deep learning for improved accuracy       |
| **NLTK Preprocessing** | Cleaning, stopword removal, tokenization  |
| **Streamlit Web App**  | Easy and interactive sentiment prediction |

---

## ⚙️ Installation

1. **Clone the repo:**

```bash
git clone https://github.com/your-username/tweet-sentiment-analyzer.git
cd tweet-sentiment-analyzer
```

2. **Create a virtual environment (Python 3.10+):**

```bash
python -m venv tweet_env
tweet_env\Scripts\activate    # Windows
# source tweet_env/bin/activate   # Mac/Linux
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Models

1. **Train Models & Save:**

```bash
python main.py
```

✅ This creates:

- models/saved_models/lstm_model.h5

- models/saved_models/tokenizer.json

- models/saved_models/logistic_model.pkl

- models/saved_models/tfidf_vectorizer.pkl

---

## 🌐 Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

- Enter a tweet in the input box

- Choose a model (default is LSTM)

- Get predicted sentiment with confidence scores

---

## 📦 Requirements

All dependencies are listed in requirements.txt. Major libraries:

- tensorflow
- scikit-learn
- nltk
- streamlit
- pandas

---

## 📊 Dataset

We use the Sentiment140 dataset (160k tweets) with pre-labeled sentiments:

- 0 → Negative
- 2 → Neutral
- 4 → Positive

Dataset link: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## 🙋‍♂️ Author

**Tilak Savani**  
Master’s in Computer Science, University of Georgia  
Domain: Artificial Intelligence & Machine Learning

---

## ⭐ Credits

- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)

---

## 📄 License

This project is open-source and available under the **MIT License**.
