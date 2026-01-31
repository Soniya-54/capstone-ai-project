import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# 1. Load
df = pd.read_csv("robust_complaints_data.csv")
df['cleaned'] = df['complaint_text'].apply(preprocess_text)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['category'], test_size=0.2)

# 3. Vectorize (Using TRIGRAMS 1-3)
# max_features=2500 gives enough memory for complex phrases
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=2500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training High-Resolution Trigram Model...")
model = CalibratedClassifierCV(LinearSVC(dual=True, C=0.8), cv=5)
model.fit(X_train_tfidf, y_train)

# 4. Save
joblib.dump(model, 'complaint_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\n--- PERFORMANCE REPORT ---")
print(classification_report(y_test, model.predict(X_test_tfidf)))