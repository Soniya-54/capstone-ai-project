import os, re, datetime, joblib, nltk
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from models import db, User, Complaint
from ai_engine import get_ai_prediction

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enterprise_level_secret_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///city_v2.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and user.password == request.form['password']:
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid Credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        if User.query.filter_by(email=email).first():
            flash('Email already exists.')
            return redirect(url_for('register'))
        new_user = User(email=email, password=request.form['password'], role='CITIZEN')
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    if current_user.role == 'ADMIN':
        # ABOVE PAR LOGIC: Admin sees their department AND things needing triage
        items = Complaint.query.filter(
            (Complaint.final_category == current_user.department) | 
            (Complaint.final_category == 'MANUAL TRIAGE')
        ).order_by(Complaint.timestamp.desc()).all()
        return render_template('admin_dashboard.html', complaints=items)
    else:
        items = Complaint.query.filter_by(citizen_id=current_user.id).all()
        return render_template('citizen_dashboard.html', complaints=items)

@app.route('/submit', methods=['POST'])
@login_required
def submit_complaint():
    text = request.form['text']
    loc = request.form['location']
    file = request.files.get('image')
    
    filename = 'none.jpg'
    if file and file.filename != '':
        filename = secure_filename(f"{current_user.id}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    category, confidence = get_ai_prediction(text)
    # Triage Rule
    final_route = category if confidence > 0.70 else "MANUAL TRIAGE"

    new_c = Complaint(
        citizen_id=current_user.id, text=text, location=loc,
        image_file=filename, ai_category=category,
        final_category=final_route, confidence=confidence
    )
    db.session.add(new_c)
    db.session.commit()
    flash(f"AI categorized this as {category} with {confidence:.1%} confidence.")
    return redirect(url_for('dashboard'))

@app.route('/resolve/<int:id>', methods=['POST'])
@login_required
def resolve_complaint(id):
    complaint = Complaint.query.get(id)
    complaint.status = 'Resolved'
    db.session.commit()
    flash(f"Case #{id} Resolved.")
    return redirect(url_for('dashboard'))

@app.route('/reassign/<int:id>', methods=['POST'])
@login_required
def reassign(id):
    complaint = Complaint.query.get(id)
    complaint.final_category = request.form['new_dept']
    db.session.commit()
    flash(f"Case #{id} re-routed to {complaint.final_category}.")
    return redirect(url_for('dashboard'))

# --- SYSTEM INITIALIZER ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create a suite of Admins for different departments
        depts = ['HEALTHCARE', 'INFRASTRUCTURE', 'SANITATION', 'PUBLIC SAFETY', 'ADMINISTRATION']
        for d in depts:
            email = f"admin_{d.lower()}@city.gov"
            if not User.query.filter_by(email=email).first():
                user = User(email=email, password='123', role='ADMIN', department=d)
                db.session.add(user)
        db.session.commit()
        print("System ready with multiple departmental admins.")
    app.run(debug=True)