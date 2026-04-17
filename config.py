"""
Configuration settings for the Financial Fraud Detection System
"""
from pathlib import Path
import os

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Model directory
MODEL_DIR = BASE_DIR / "model" / "project3"

# Database
DATABASE_URI = 'sqlite:///fraud_detection.db'

# Flask settings
SECRET_KEY = "financial_fraud_detection_2024"
DEBUG = True
TEMPLATES_AUTO_RELOAD = True

# JSON settings
JSON_SORT_KEYS = False
JSONIFY_PRETTYPRINT_REGULAR = True

# Chart settings
CHARTS_COLORS = {
    'danger': '#ef5350',
    'success': '#66bb6a',
    'warning': '#ffa726',
    'info': '#29b6f6',
    'primary': '#667eea',
}

# Demo users
DEMO_USERS = {
    "admin": {"password": "demo123", "role": "Admin"},
    "analyst": {"password": "demo123", "role": "Analyst"},
    "reviewer": {"password": "demo123", "role": "Reviewer"}
}

# Prediction thresholds
FRAUD_THRESHOLDS = {
    'high_amount': 10000,
    'night_hours': [(20, 23), (0, 6)],
}

LOAN_THRESHOLDS = {
    'min_credit_score': 300,
    'max_credit_score': 850,
    'min_age': 18,
}

RISK_THRESHOLDS = {
    'high_risk': 60,
    'medium_risk': 35,
    'low_risk': 0,
}

# CSV Upload
CSV_ALLOWED_EXTENSIONS = {'csv'}
CSV_MAX_SIZE = 5 * 1024 * 1024  # 5MB

# Pagination
ITEMS_PER_PAGE = 10

# Report settings
REPORT_TITLE = "Financial Fraud Detection Report"
REPORT_AUTHOR = "AI Security System"
