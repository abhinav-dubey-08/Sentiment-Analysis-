# 🧠 Sentiment Analysis Web App (Streamlit + Flask)

A full-stack **Machine Learning Web App** to predict the **sentiment (Positive / Negative)** of text. This project uses **XGBoost**, **NLTK**, and **CountVectorizer** for the model, and features:

- 📄 CSV-based bulk predictions
- 💬 Real-time single-sentence predictions
- 📊 Sentiment distribution pie chart
- 📥 CSV download of prediction results

> Frontend: `Streamlit` | Backend: `Flask` | Model: `XGBoost Classifier`

---

## 📸 Demo

| 💬 Single Sentence Prediction | 📄 Bulk CSV Upload & Graph |
|------------------------------|----------------------------|
| ![Single Prediction](screenshots/single_prediction.png) | ![Bulk Prediction](screenshots/bulk_prediction.png) |

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

