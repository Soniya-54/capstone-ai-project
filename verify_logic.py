import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure resources are available
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Setup Preprocessing (Must match training exactly!)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# 2. Load the Brain
try:
    model = joblib.load('complaint_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    print("--- AI Brain Loaded Successfully ---\n")
except FileNotFoundError:
    print("ERROR: Could not find model files. Did you run train_model.py first?")
    exit()

# 3. Define Test Scenarios (Real-world "Stress Test" cases)
test_cases = [
    # Infrastructure Test
    "While I was going through the downtown area, I noticed a massive pothole that caused my car to swerve dangerously. I almost ran to death. That was fuckin frustrating. Idk what to do.",
    
    # Healthcare Test
    "I lost my consciousness and banged my head in the floor. I was rushed to the city."
    "There is a pile of trash and rotting food on the sidewalk and it smells terrible.",
    
    # Administration Test
    "The clerk at the city hall refused to help me with my birth certificate and asked for money.",
    
    # Public Safety Test
    "A gang of people were fighting when I was returning to my house. I almost got hurt in the cross-fire.",
    
    # General/Noise Test
    "The sky is beautiful",

    # Healthcare
    "My brother was puking so much. Still nobody came to attend me. ",
    
    # Ambiguity Stress Test (Cross-Category)
    "The bus where patient was there couldn't get through the street because of the huge pothole and trash."
]

print(f"{'INPUT TEXT':<60} | {'PREDICTION':<20} | {'CONFIDENCE'}")
print("-" * 100)

for text in test_cases:
    # Preprocess
    cleaned = clean_text(text)
    # Vectorize
    vec = tfidf.transform([cleaned])
    # Predict
    probs = model.predict_proba(vec)[0]
    max_idx = probs.argmax()
    category = model.classes_[max_idx]
    confidence = probs[max_idx]
    
    # Display results
    short_text = (text[:57] + '..') if len(text) > 57 else text
    print(f"{short_text:<60} | {category:<20} | {confidence:.2%}")

# Threshold Analysis
print("\n--- PROFESSOR'S CRITICAL ANALYSIS ---")
print("Target Threshold for Triage: 0.70 (70%)")
print("Look at the 'Ambiguity Stress Test' above. If the score is low (e.g. < 70%),")
print("it proves your system correctly identifies complex multi-category issues.")