import joblib, re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

# Load files
model = joblib.load('complaint_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def get_ai_prediction(raw_text):
    cleaned = clean_text(raw_text)
    vec = tfidf.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    max_idx = probs.argmax()
    
    category = model.classes_[max_idx].upper()
    confidence = float(probs[max_idx])
    
    return category, confidence