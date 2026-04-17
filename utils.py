"""
Utility functions for the Financial Fraud Detection System
"""
import json
from functools import wraps
from flask import session, flash, redirect
import pandas as pd
import numpy as np

def safe_float(val):
    """Safely convert value to float"""
    try:
        return float(val) if val not in [None, ""] else 0.0
    except (ValueError, TypeError):
        return 0.0

def safe_int(val):
    """Safely convert value to integer"""
    try:
        return int(val) if val not in [None, ""] else 0
    except (ValueError, TypeError):
        return 0

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("⚠️ Please login first", "warning")
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def save_transaction_to_db(db, Transaction, user_id, trans_type, amount, location, device, result, confidence, details=None):
    """Save transaction to database"""
    try:
        trans = Transaction(
            user_id=user_id,
            trans_type=trans_type,
            amount=amount,
            location=location,
            device=device,
            result=result,
            confidence=confidence,
            details=json.dumps(details) if details else "{}"
        )
        db.session.add(trans)
        db.session.commit()
        return trans
    except Exception as e:
        db.session.rollback()
        print(f"Error saving transaction: {e}")
        return None

def validate_csv_data(row, required_fields):
    """Validate CSV row data"""
    for field in required_fields:
        if field not in row or row[field] == '':
            return False, f"Missing required field: {field}"
    return True, "Valid"

def calculate_fraud_features(amount, hour, device, location):
    """Calculate features for fraud detection"""
    is_night = 1 if (hour >= 20 or hour <= 6) else 0
    is_high = 1 if amount > 10000 else 0
    new_device = 1 if device.lower() not in ["mobile", "phone"] else 0
    location_change = 1 if location.lower() not in ["delhi", "home"] else 0
    
    return [amount, is_night, is_high, new_device, location_change]

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1f}%"

def get_result_badge_class(result):
    """Get badge CSS class based on result"""
    if 'Fraud' in result or 'Rejected' in result or 'DEFAULT' in result or 'High' in result:
        return 'badge-danger'
    elif 'Safe' in result or 'Approved' in result or 'Low' in result:
        return 'badge-success'
    elif 'Medium' in result:
        return 'badge-warning'
    else:
        return 'badge-info'

def get_result_icon(result):
    """Get icon for result"""
    if 'Fraud' in result or 'Rejected' in result:
        return '⚠️'
    elif 'Safe' in result or 'Approved' in result:
        return '✅'
    elif 'Type' in result or 'Medium' in result:
        return '⚡'
    elif 'Anomaly' in result:
        return '🚨'
    else:
        return '📊'

class MLModelManager:
    """Manages machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.metadata = {}

    def load_model(self, model_path, meta_path):
        """Load a model and its metadata"""
        import joblib
        try:
            model = joblib.load(model_path)
            with open(meta_path) as f:
                meta = json.load(f)
            return model, meta
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None, None

    def predict_fraud(self, model, features):
        """Make fraud prediction"""
        try:
            pred = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            classes = list(model.classes_)
            if 1 in classes:
                prob = proba[classes.index(1)] * 100
            else:
                prob = max(proba) * 100
            return pred, round(prob, 2)
        except Exception as e:
            print(f"Error in fraud prediction: {e}")
            return None, None

    def predict_loan(self, model, data):
        """Make loan prediction"""
        try:
            pred = model.predict(data)[0]
            proba = model.predict_proba(data)[0]
            classes = list(model.classes_)
            if pred in classes:
                prob = proba[classes.index(pred)] * 100
            else:
                prob = max(proba) * 100
            return pred, round(prob, 2)
        except Exception as e:
            print(f"Error in loan prediction: {e}")
            return None, None

class DataProcessor:
    """Processes and validates data"""
    
    @staticmethod
    def prepare_fraud_data(amount, hour, device, location, model_meta):
        """Prepare data for fraud prediction"""
        try:
            features = calculate_fraud_features(amount, hour, device, location)
            df = pd.DataFrame([features], columns=model_meta["feature_names"])
            return df, True
        except Exception as e:
            print(f"Error preparing fraud data: {e}")
            return None, False

    @staticmethod
    def prepare_loan_data(age, income, loan_amount, credit, employment):
        """Prepare data for loan prediction"""
        try:
            data = pd.DataFrame({
                "person_age": [safe_float(age)],
                "person_income": [safe_float(income)],
                "person_emp_exp": [safe_float(employment)],
                "loan_amnt": [safe_float(loan_amount)],
                "loan_int_rate": [5.0],
                "loan_percent_income": [safe_float(loan_amount) / safe_float(income) if safe_float(income) > 0 else 0.5],
                "cb_person_cred_hist_length": [3.0],
                "credit_score": [safe_float(credit)],
                "person_gender": ["M"],
                "person_education": ["High School"],
                "person_home_ownership": ["RENT"],
                "loan_intent": ["PERSONAL"],
                "previous_loan_defaults_on_file": ["No"],
            })
            return data, True
        except Exception as e:
            print(f"Error preparing loan data: {e}")
            return None, False

    @staticmethod
    def generate_loan_reasons(age, income, loan_amount, credit, employment):
        """Generate human-readable loan risk reasons."""
        reasons = []
        if income <= 0:
            reasons.append("Income value is missing or invalid")
        else:
            ratio = loan_amount / income
            if ratio > 1:
                reasons.append("Loan amount is greater than income")
            elif ratio > 0.5:
                reasons.append("Loan exceeds 50% of income")
            elif ratio > 0.3:
                reasons.append("Loan is high compared to income")
            else:
                reasons.append("Loan is within a reasonable income ratio")

        if credit <= 0:
            reasons.append("Credit score is missing")
        elif credit < 600:
            reasons.append("Low credit score")
        elif credit < 700:
            reasons.append("Moderate credit score")
        elif credit < 750:
            reasons.append("Good credit score")
        else:
            reasons.append("Strong credit score")

        if employment < 2:
            reasons.append("Short employment history")
        elif employment < 5:
            reasons.append("Moderate employment history")
        else:
            reasons.append("Stable employment history")

        return reasons

    @staticmethod
    def calculate_risk_score(age, income, loan_amount, credit):
        """Calculate risk score"""
        risk_score = 0
        
        # Credit score risk
        if credit < 600:
            risk_score += 40
        elif credit < 700:
            risk_score += 25
        elif credit < 750:
            risk_score += 10
        
        # Debt ratio risk
        if income > 0:
            debt_ratio = loan_amount / income
            if debt_ratio > 0.5:
                risk_score += 35
            elif debt_ratio > 0.3:
                risk_score += 20
        
        # Age risk
        if age < 25:
            risk_score += 15
        elif age > 65:
            risk_score += 10
        
        return min(100, risk_score)

    @staticmethod
    def get_risk_level(score):
        """Get risk level from score"""
        if score >= 60:
            return "🔴 HIGH RISK"
        elif score >= 35:
            return "🟠 MEDIUM RISK"
        else:
            return "🟢 LOW RISK"
