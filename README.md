# 🧠 Sentiment Analysis Web App (Streamlit + Flask)

A full-stack **Machine Learning Web App** to predict the **sentiment (Positive / Negative)** of text. This project uses **XGBoost**, **NLTK**, and **CountVectorizer** for the model, and features:

- 📄 CSV-based bulk predictions
- 💬 Real-time single-sentence predictions
- 📊 Sentiment distribution pie chart
- 📥 CSV download of prediction results

> Frontend: `Streamlit` | Backend: `Flask` | Model: `XGBoost Classifier`

---


---

## 🧠 Model Details

- **Model:** XGBoost Classifier
- **Preprocessing:** NLTK stopwords, Porter Stemmer
- **Vectorization:** CountVectorizer
- **Feature Scaling:** StandardScaler

Files:
- `Models/model_xgb.pkl` – Trained sentiment model
- `Models/scaler.pkl` – StandardScaler object
- `Models/countVectorizer.pkl` – CountVectorizer object

---

## 📂 Project Structure
```
sentiment-analysis-app/
├── Models/                       # Contains saved ML model & preprocessing files
│   ├── model_xgb.pkl             # Trained XGBoost classifier
│   ├── scaler.pkl                # Feature scaler used during training
│   └── countVectorizer.pkl       # CountVectorizer for text tokenization
│
├── templates/
│   └── landing.html              # Optional Flask-based HTML frontend
│
├── main.py                        # Streamlit frontend (main UI)
├── api.py                        # Flask backend API for predictions

