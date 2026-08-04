import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from preprocess import preprocess_text

df = pd.read_csv('dataset.csv')

df['processed_text'] = df['complaint_text'].apply(preprocess_text)

X = df['processed_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

evaluate_model(y_test, nb_pred, "Naive Bayes")
evaluate_model(y_test, lr_pred, "Logistic Regression")

joblib.dump(lr_model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


print("\n✓ Model and vectorizer saved successfully!")
