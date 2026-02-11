# 📧 Intelligent Spam Email Classifier

A high-performance machine learning project that uses the **Naïve Bayes Algorithm** to filter spam emails from legitimate ones.

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-Multinomial_Naïve_Bayes-orange)

## 📌 Project Overview
This system is designed to automatically detect and filter spam emails. It uses Natural Language Processing (NLP) techniques for text preprocessing and a Naïve Bayes classifier for accurate prediction.

### ✨ Key Features
- **Real-Time Detection**: Instantly classifies emails as Spam (Red) or Safe (Green).
- **High Accuracy**: Typically achieves ~98% accuracy on standard datasets.
- **Premium UI**: Modern, responsive interface built with HTML5, CSS3, and JavaScript.
- **FastAPI Backend**: Lightweight and fast Python backend.

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8 or higher installed.

### 1️⃣ Setup Environment
Open your terminal in the project folder and install dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model
Before running the app, you need to train the model on the dataset:
```bash
python model/train_model.py
```
*This will create `classifier.pkl` and `vectorizer.pkl` in the `model` folder.*

### 3️⃣ Start the Server
Run the FastAPI server:
```bash
uvicorn main:app --reload
```

### 4️⃣ Use the App
Open your browser and go to:
👉 **http://127.0.0.1:8000**

---

## 📂 Project Structure
```
spam_classifier/
├── data/               # Contains dataset (spam.csv / SMSSpamCollection)
├── model/              # 
│   ├── train_model.py  # Script to train the ML model
│   ├── classifier.pkl  # Saved model file
│   └── vectorizer.pkl  # Saved vectorizer file
├── static/             # Frontend Assets
│   ├── index.html      # Main UI
│   ├── style.css       # Premium Styling
│   └── script.js       # API Interaction Logic
├── main.py             # FastAPI Backend Server
├── requirements.txt    # List of dependencies
└── README.md           # This file
```

## 🧠 Algorithm Used
**Multinomial Naïve Bayes**: Chosen for its superior performance in text classification tasks. It calculates the probability of an email being spam based on the frequency of words it contains.

## 👥 Authors
- **CSE Final Year Student**
