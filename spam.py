import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords (only first time)
nltk.download('stopwords')

# Load dataset (UCI dataset format - tab separated)
data = pd.read_csv("spam.csv", sep='\t', header=None, names=['label','message'])

# Convert labels to numbers
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

data['message'] = data['message'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

# Accuracy
nb_accuracy = accuracy_score(y_test, nb_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

# Confusion Matrix
print("\nConfusion Matrix (Naive Bayes):")
print(confusion_matrix(y_test, nb_pred))

# Classification Report
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, nb_pred))

# Accuracy Graph
plt.figure()
plt.bar(["Naive Bayes", "Logistic Regression"], [nb_accuracy, lr_accuracy])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.9, 1.0)
plt.show()

# Top 10 Spam Indicative Words
feature_names = vectorizer.get_feature_names_out()
spam_prob = nb_model.feature_log_prob_[1]

top_spam_words = sorted(zip(spam_prob, feature_names), reverse=True)[:10]

print("\nTop 10 Spam Indicative Words:")
for prob, word in top_spam_words:
    print(word)

# Custom Email Testing
sample = ["Congratulations! You have won a free lottery ticket. Click now!"]
sample_clean = [clean_text(sample[0])]
sample_tfidf = vectorizer.transform(sample_clean)

prediction = nb_model.predict(sample_tfidf)

if prediction[0] == 1:
    print("\nThe sample email is SPAM")
else:
    print("\nThe sample email is NOT SPAM")
