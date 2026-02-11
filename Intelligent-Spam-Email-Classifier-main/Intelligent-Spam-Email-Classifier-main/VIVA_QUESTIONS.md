# 🎓 Spam Classifier - Viva Questions & Answers

## 🔹 Core Concept Questions

**Q1: Why did you choose Naïve Bayes for this project?**
> **Answer**: Naïve Bayes is the "Gold Standard" for text classification. It is:
> 1. **Fast**: Training and prediction are extremely quick.
> 2. **Effective**: It works surprisingly well with high-dimensional data like text (Bag of Words).
> 3. **Simple**: It assumes independence between words, which simplifies computation without sacrificing much accuracy for spam detection.

**Q2: What is the "Naïve" assumption in Naïve Bayes?**
> **Answer**: It assumes that the occurrence of one word is **independent** of the occurrence of another. For example, the probability of seeing "Free" is calculated independently of seeing "iPhone", even though they often appear together in spam. Despite this "naïve" assumption, it works very well in practice.

**Q3: Which variant of Naïve Bayes did you use and why?**
> **Answer**: I used **Multinomial Naïve Bayes (`MultinomialNB`)**.
> - Gaussian NB is for continuous data (like height/weight).
> - **Multinomial NB** is specifically designed for **discrete counts**, such as word counts in text classification, making it ideal for our Bag of Words approach.

---

## 🔹 Technical Implementation Questions

**Q4: How did you convert text into numbers for the model?**
> **Answer**: I used **CountVectorizer** (Bag of Words model).
> 1. It creates a vocabulary of all unique words in the dataset.
> 2. It converts each email into a vector of numbers, where each number represents the frequency of a word from the vocabulary in that email.

**Q5: What is Data Preprocessing and what steps did you take?**
> **Answer**: Preprocessing cleans the raw data to improve model quality. I used:
> 1. **Lowercasing**: To treat "Spam" and "spam" as the same word.
> 2. **Punctuation Removal**: Removing `!`, `.`, `,` as they usually don't carry class information.
> 3. **Unicode Handling**: Ensuring special characters don't crash the model.

**Q6: What is the accuracy of your model?**
> **Answer**: The model achieved an accuracy of **98.21%** on the test set.

**Q7: How does the Backend communicate with the Frontend?**
> **Answer**:
> - **Frontend**: Uses JavaScript `fetch()` to send a POST request with the email text as JSON.
> - **Backend**: Built with **FastAPI**. It receives the JSON, runs it through the saved model (`classifier.pkl`), and returns the prediction ("Spam" or "Ham") and confidence score.

---

## 🔹 Tricky Questions (The "Curveballs")

**Q8: What happens if the model encounters a word it has never seen before?**
> **Answer**: This is the "Zero Frequency" problem. `CountVectorizer` simply ignores words not in its training vocabulary. In probability calculations, we use **Laplace Smoothing** (alpha parameter in Sklearn) to ensure the probability doesn't become zero, preventing the whole calculation from collapsing.

**Q9: Why didn't you use Deep Learning (RNN/LSTM/BERT)?**
> **Answer**:
> 1. **Overkill**: For simple spam detection, Deep Learning is computationally expensive and slow.
> 2. **Data Size**: We have a small dataset (~5000 emails). Deep Learning requires massive datasets to outperform Naïve Bayes significantly.
> 3. **Efficiency**: Naïve Bayes provides real-time results with minimal CPU usage, which is better for a lightweight web service.

**Q10: What are False Positives and False Negatives in this context?**
> - **False Positive**: A legitimate email (Ham) classified as Spam. (Very bad, user misses important mail).
> - **False Negative**: A Spam email classified as Ham. (Annoying, but less critical).
> *My model prioritizes high precision to minimize False Positives.*

---

## 🔹 Future Scope

**Q11: How would you improve this project further?**
> **Answer**:
> 1. Use **TF-IDF** instead of CountVectorizer to weigh unique words higher.
> 2. Implement **N-grams** (looking at pairs of words like "credit card" instead of just "credit" and "card").
> 3. Add a **Feedback Loop** where users can mark misclassified emails to retrain the model dynamically.
