from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), default='CITIZEN') # CITIZEN or ADMIN
    department = db.Column(db.String(50), nullable=True) # For Admins: e.g., 'HEALTHCARE'

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    citizen_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    text = db.Column(db.Text, nullable=False)
    image_file = db.Column(db.String(100), default='none.jpg')
    location = db.Column(db.String(100))
    ai_category = db.Column(db.String(50))
    final_category = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    status = db.Column(db.String(20), default='Pending') # Pending, In-Progress, Resolved
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)