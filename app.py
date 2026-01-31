from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import joblib
import datetime
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Standardized NLP setup
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = "soniya_capstone_2026_secured"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///city_complaints.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- AI INTEGRATION LAYER ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# Load the Brain
try:
    model = joblib.load('complaint_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    print("AI Brain integrated successfully.")
except:
    print("CRITICAL: AI Model files missing!")

# --- DATABASE SCHEMA ---
class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    raw_text = db.Column(db.Text, nullable=False)
    ai_category = db.Column(db.String(50))
    final_category = db.Column(db.String(50)) 
    confidence_score = db.Column(db.Float)
    status = db.Column(db.String(30), default="AI DISPATCHED")
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def citizen_portal():
    if request.method == 'POST':
        raw_text = request.form['complaint_text']
        cleaned = clean_text(raw_text)
        vec = tfidf.transform([cleaned])
        
        probs = model.predict_proba(vec)[0]
        max_idx = probs.argmax()
        category = model.classes_[max_idx]
        confidence = probs[max_idx]
        
        # TRIAGE LOGIC: If AI is unsure, route to Manual Triage
        # This addresses your research point on 'Accountability'
        if confidence < 0.70:
            assigned_route = "MANUAL HUMAN TRIAGE"
            current_status = "PENDING REVIEW"
        else:
            assigned_route = category
            current_status = "AI DISPATCHED"

        new_complaint = Complaint(
            raw_text=raw_text,
            ai_category=category,
            final_category=assigned_route,
            confidence_score=confidence,
            status=current_status
        )
        db.session.add(new_complaint)
        db.session.commit()
        
        flash(f"Submission recorded! Tracking ID: #{new_complaint.id}.")
        return redirect(url_for('citizen_portal'))

    return render_template('index.html')

@app.route('/admin')
def admin_dashboard():
    complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    return render_template('admin.html', complaints=complaints)

# NEW ROUTE: Manual Override Logic
@app.route('/override/<int:id>', methods=['POST'])
def override_route(id):
    complaint = Complaint.query.get(id)
    if not complaint:
        return "Error: Complaint not found", 404
        
    new_dept = request.form['new_dept']
    complaint.final_category = new_dept
    complaint.status = "HUMAN VERIFIED"
    db.session.commit()
    
    flash(f"Complaint #{id} manually re-routed to {new_dept}.")
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)