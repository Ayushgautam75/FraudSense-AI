"""
AI Financial Fraud Detection System - Main Application
=====================================================
Professional Flask-based web application for real-time financial fraud detection,
loan default prediction, and risk assessment using Machine Learning algorithms.

Author: AI Security Team
Version: 2.0 (Professional Edition)
License: MIT
"""

from flask import Flask, render_template, request, redirect, session, flash, send_file
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import json
import csv
import io

# Import from local modules
from config import (
    BASE_DIR, MODEL_DIR, DATABASE_URI, SECRET_KEY, DEMO_USERS, 
    FRAUD_THRESHOLDS, LOAN_THRESHOLDS, RISK_THRESHOLDS
)
from models import db, Transaction
from utils import (
    safe_float, safe_int, login_required, save_transaction_to_db,
    DataProcessor, MLModelManager
)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.secret_key = SECRET_KEY

# Initialize database
db.init_app(app)

# Initialize ML Model Manager
ml_manager = MLModelManager()

# ============ LOAD ML MODELS ============
def load_all_models():
    """Load all machine learning models and metadata"""
    try:
        print(f"📁 Looking for models in: {MODEL_DIR}")
        
        # Credit Card Fraud model
        cc_rf, cc_meta = ml_manager.load_model(
            MODEL_DIR / "cc_fraud_rf.pkl",
            MODEL_DIR / "cc_meta.json"
        )
        if cc_rf is None:
            raise Exception("cc_fraud_rf model failed to load")
        print("✅ CC Fraud model loaded")
        
        # Loan Default model
        loan_pipe, loan_meta = ml_manager.load_model(
            MODEL_DIR / "loan_default_rf.pkl",
            MODEL_DIR / "loan_meta.json"
        )
        if loan_pipe is None:
            raise Exception("loan_default_rf model failed to load")
        print("✅ Loan model loaded")
        
        # Isolation Forest (Anomaly detection)
        iso_forest, iso_meta = ml_manager.load_model(
            MODEL_DIR / "iso_forest.pkl",
            MODEL_DIR / "iso_meta.json"
        )
        if iso_forest is None:
            raise Exception("iso_forest model failed to load")
        print("✅ Anomaly model loaded")
        
        iso_scaler, _ = ml_manager.load_model(
            MODEL_DIR / "iso_scaler.pkl",
            MODEL_DIR / "iso_meta.json"
        )
        print("✅ Anomaly scaler loaded")
        
        # K-Means (Spending patterns)
        spend_km, spend_meta = ml_manager.load_model(
            MODEL_DIR / "spend_kmeans.pkl",
            MODEL_DIR / "spend_meta.json"
        )
        if spend_km is None:
            raise Exception("spend_kmeans model failed to load")
        print("✅ Spending model loaded")
        
        spend_scaler, _ = ml_manager.load_model(
            MODEL_DIR / "spend_scaler.pkl",
            MODEL_DIR / "spend_meta.json"
        )
        print("✅ Spending scaler loaded")
        
        print("✅ All models loaded successfully")
        return {
            'cc_rf': cc_rf,
            'cc_meta': cc_meta,
            'loan_pipe': loan_pipe,
            'loan_meta': loan_meta,
            'iso_forest': iso_forest,
            'iso_meta': iso_meta,
            'iso_scaler': iso_scaler,
            'spend_km': spend_km,
            'spend_meta': spend_meta,
            'spend_scaler': spend_scaler
        }
    except Exception as e:
        print(f"❌ ERROR Loading Models: {e}")
        raise RuntimeError(f"Failed to load ML models: {e}")

# Load models
MODELS = load_all_models()

# ============ AUTHENTICATION ROUTES ============
@app.route("/", methods=["GET"])
def index():
    """Redirect to dashboard if logged in, else to login"""
    if 'user' in session:
        return redirect("/dashboard")
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page and authentication"""
    if request.method == "POST":
        username = request.form.get("user", "").strip()
        password = request.form.get("pwd", "").strip()

        if username in DEMO_USERS and DEMO_USERS[username]["password"] == password:
            session['user'] = username
            session['role'] = DEMO_USERS[username]["role"]
            flash(f"✅ Welcome {username}!", "success")
            print(f"🔐 User logged in: {username}")
            return redirect("/dashboard")
        else:
            flash("❌ Invalid credentials. Try admin / demo123", "danger")
            return render_template("login_new.html", error="Invalid username or password")

    return render_template("login_new.html")

@app.route("/logout", methods=["POST"])
def logout():
    """User logout and session clear"""
    username = session.get('user', 'Unknown')
    session.clear()
    flash("✅ Logged out successfully", "success")
    print(f"🚪 User logged out: {username}")
    return redirect("/login")

# ============ MAIN PAGES ============
@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    """Main dashboard with prediction modules"""
    user = session.get('user')
    today = datetime.now().date()
    start_date = today - timedelta(days=6)

    transactions = Transaction.query.filter_by(user_id=user).order_by(Transaction.date.desc()).all()
    stats = Transaction.get_user_statistics(user)

    day_totals = {}
    for trans in transactions:
        day_key = trans.date.date()
        if day_key not in day_totals:
            day_totals[day_key] = {'fraud': 0, 'legit': 0, 'total': 0}
        if 'Fraud' in trans.result:
            day_totals[day_key]['fraud'] += 1
        if 'Safe' in trans.result:
            day_totals[day_key]['legit'] += 1
        day_totals[day_key]['total'] += 1

    labels = []
    fraud_trend = []
    legit_trend = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        labels.append(day.strftime('%a'))
        fraud_trend.append(day_totals.get(day, {}).get('fraud', 0))
        legit_trend.append(day_totals.get(day, {}).get('legit', 0))

    fraud_total = stats.get('fraud_detected', 0)
    safe_total = stats.get('safe', 0)
    total_transactions = stats.get('total', 0)
    fraud_rate = round((fraud_total / total_transactions) * 100, 2) if total_transactions else 0
    safe_rate = round((safe_total / total_transactions) * 100, 2) if total_transactions else 0

    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)
    yesterday_total = day_totals.get(yesterday, {}).get('total', 0)
    before_total = day_totals.get(day_before, {}).get('total', 0)
    total_change = round(((yesterday_total - before_total) / before_total) * 100, 1) if before_total else (100 if yesterday_total else 0)

    yesterday_fraud = day_totals.get(yesterday, {}).get('fraud', 0)
    before_fraud = day_totals.get(day_before, {}).get('fraud', 0)
    fraud_change = round(((yesterday_fraud - before_fraud) / before_fraud) * 100, 1) if before_fraud else (100 if yesterday_fraud else 0)

    yesterday_safe = day_totals.get(yesterday, {}).get('legit', 0)
    before_safe = day_totals.get(day_before, {}).get('legit', 0)
    safe_change = round(((yesterday_safe - before_safe) / before_safe) * 100, 1) if before_safe else (100 if yesterday_safe else 0)

    recent_transactions = [trans.to_dict() for trans in transactions[:5]]
    last_transaction = recent_transactions[0] if recent_transactions else None

    banner_message = (
        f"🚨 {fraud_total} fraud alerts detected in the last 7 days!" if fraud_total else "✅ No fraud alerts detected in the last 7 days."
    )

    latest_reasons = []
    if last_transaction:
        details = last_transaction.get('details', {})
        if isinstance(details, dict):
            if 'reasons' in details:
                latest_reasons = details['reasons'] if isinstance(details['reasons'], list) else [details['reasons']]
            elif last_transaction.get('trans_type') == 'loan':
                ratio = details.get('loan_ratio')
                if isinstance(ratio, (int, float)):
                    if ratio > 1:
                        latest_reasons.append('Loan amount exceeds income')
                    elif ratio > 0.5:
                        latest_reasons.append('Loan is more than 50% of income')
                    elif ratio > 0.3:
                        latest_reasons.append('Loan is high relative to income')
                    else:
                        latest_reasons.append('Loan is within a reasonable range for income')
                credit_score = details.get('credit_score')
                if isinstance(credit_score, (int, float)):
                    if credit_score < 600:
                        latest_reasons.append('Credit score is low')
                    elif credit_score < 700:
                        latest_reasons.append('Credit score is moderate')
                    elif credit_score < 750:
                        latest_reasons.append('Credit score is good')
                    else:
                        latest_reasons.append('Credit score is strong')
            elif last_transaction.get('trans_type') == 'fraud':
                if last_transaction['amount'] > 10000:
                    latest_reasons.append('High Amount')
                hour_value = details.get('hour')
                if isinstance(hour_value, (int, float)) and (hour_value >= 20 or hour_value <= 6):
                    latest_reasons.append('Night Time Transaction')
                transaction_type = details.get('transaction_type')
                if isinstance(transaction_type, str) and transaction_type.strip():
                    latest_reasons.append(f"Type: {transaction_type.strip().capitalize()}")

    spark_total = [fraud_trend[i] + legit_trend[i] for i in range(len(labels))]
    spark_rate = [
        round((fraud_trend[i] / (fraud_trend[i] + legit_trend[i])) * 100, 1)
        if (fraud_trend[i] + legit_trend[i]) else 0
        for i in range(len(labels))
    ]

    return render_template(
        "dashboard_new.html",
        current_date=datetime.now().strftime("%A, %B %d, %Y"),
        user=user,
        role=session.get('role'),
        stats=stats,
        fraud_rate=fraud_rate,
        safe_rate=safe_rate,
        fraud_total=fraud_total,
        safe_total=safe_total,
        line_labels=labels,
        line_fraud=fraud_trend,
        line_legit=legit_trend,
        spark_total=spark_total,
        spark_rate=spark_rate,
        pie_data=[fraud_total, safe_total],
        recent_transactions=recent_transactions,
        last_transaction=last_transaction,
        last_reasons=latest_reasons,
        banner_message=banner_message,
        total_change=total_change,
        fraud_change=fraud_change,
        safe_change=safe_change
    )

@app.route("/history", methods=["GET"])
@login_required
def history():
    """Transaction history and analytics"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10

        # Base query
        query = Transaction.query.filter_by(user_id=session['user']).order_by(Transaction.date.desc())

        # Apply filters
        search = request.args.get('search', '').strip()
        if search:
            query = query.filter(
                (Transaction.location.ilike(f'%{search}%')) |
                (Transaction.device.ilike(f'%{search}%'))
            )

        trans_type = request.args.get('type', '').strip()
        if trans_type:
            query = query.filter_by(trans_type=trans_type)

        # Pagination
        paginated = query.paginate(page=page, per_page=per_page)

        # Statistics
        all_trans = Transaction.query.filter_by(user_id=session['user']).all()
        fraud_count = len([t for t in all_trans if 'Fraud' in t.result])
        safe_count = len([t for t in all_trans if 'Safe' in t.result])
        total_count = len(all_trans)
        avg_confidence = int(np.mean([t.confidence for t in all_trans])) if all_trans else 0

        return render_template(
            "history.html",
            transactions=paginated.items,
            page=page,
            total_pages=paginated.pages,
            fraud_count=fraud_count,
            safe_count=safe_count,
            total_count=total_count,
            avg_confidence=avg_confidence
        )
    except Exception as e:
        flash(f"❌ Error loading history: {str(e)}", "danger")
        return redirect("/dashboard")

# ============ PREDICTION ROUTES ============
@app.route("/predict", methods=["POST"])
@login_required
def predict_fraud():
    """Fraud detection prediction"""
    try:
        # Safety check: Models loaded?
        if not MODELS or MODELS['cc_rf'] is None or MODELS['cc_meta'] is None:
            flash("❌ ML Models not loaded. Server error.", "danger")
            return redirect("/dashboard")
        
        # Extract form data
        amount = safe_float(request.form.get("amount"))
        hour = safe_int(request.form.get("time"))
        device = request.form.get("device", "mobile").strip()
        location = request.form.get("location", "Unknown").strip()
        trans_type = request.form.get("trans_type", "unknown").strip()

        # Safety check: Metadata?
        if "feature_names" not in MODELS['cc_meta']:
            flash("❌ Model metadata missing.", "danger")
            return redirect("/dashboard")

        # Prepare data
        from utils import calculate_fraud_features
        features = calculate_fraud_features(amount, hour, device, location)
        df = pd.DataFrame([features], columns=MODELS['cc_meta']["feature_names"])

        # Make prediction
        pred, prob = ml_manager.predict_fraud(MODELS['cc_rf'], features)
        confidence = round(prob, 2) if prob is not None else 0.0

        result = "⚠️ Fraud Detected" if pred == 1 else "✅ Transaction Safe"

        # Save to database
        save_transaction_to_db(
            db, Transaction,
            user_id=session['user'],
            trans_type='fraud',
            amount=amount,
            location=location,
            device=device,
            result=result,
            confidence=confidence,
            details={'hour': hour, 'transaction_type': trans_type}
        )

        flash_type = "danger" if pred == 1 else "success"
        flash(f"{result} (Confidence: {confidence}%)", flash_type)
        print(f"🔍 Fraud Prediction: {result} ({confidence}%) - {session['user']}")

    except Exception as e:
        flash(f"❌ Prediction Error: {str(e)}", "danger")
        print(f"❌ Error in fraud prediction: {e}")

    return redirect("/dashboard")

@app.route("/loan", methods=["POST"])
@login_required
def predict_loan():
    """Loan default prediction"""
    try:
        # Safety check: Models loaded?
        if not MODELS or MODELS['loan_pipe'] is None:
            flash("❌ ML Models not loaded. Server error.", "danger")
            return redirect("/dashboard")

        # Extract form data
        age = safe_float(request.form.get("age"))
        income = safe_float(request.form.get("income"))
        loan_amount = safe_float(request.form.get("loan"))
        credit = safe_float(request.form.get("credit"))
        employment = safe_float(request.form.get("employment", 0))

        # Prepare data
        data = DataProcessor.prepare_loan_data(age, income, loan_amount, credit, employment)[0]

        # Make prediction
        pred, confidence = ml_manager.predict_loan(MODELS['loan_pipe'], data)
        confidence = round(confidence, 2) if confidence is not None else 0.0

        result = "✅ Safe to Approve" if pred == 0 else "⚠️ Default Risk Detected"
        location = request.form.get("location", "Unknown").strip()

        loan_ratio = loan_amount / income if income > 0 else 0
        reasons = DataProcessor.generate_loan_reasons(age, income, loan_amount, credit, employment)

        # Save to database
        save_transaction_to_db(
            db, Transaction,
            user_id=session['user'],
            trans_type='loan',
            amount=loan_amount,
            location=location,
            device='loan_app',
            result=result,
            confidence=confidence,
            details={
                'age': age,
                'credit_score': credit,
                'employment': employment,
                'loan_ratio': round(loan_ratio, 2),
                'reasons': reasons
            }
        )

        flash_type = "danger" if pred == 1 else "success"
        reason_summary = reasons[0] if reasons else "Loan prediction completed"
        flash(f"{result} (Confidence: {confidence}%) — {reason_summary}", flash_type)
        print(f"💰 Loan Prediction: {result} ({confidence}%) — {reason_summary} - {session['user']}")

    except Exception as e:
        flash(f"❌ Prediction Error: {str(e)}", "danger")
        print(f"❌ Error in loan prediction: {e}")

    return redirect("/dashboard")

@app.route("/risk", methods=["POST"])
@login_required
def calculate_risk():
    """Risk score calculation"""
    try:
        age = safe_float(request.form.get("age"))
        income = safe_float(request.form.get("income"))
        loan_amount = safe_float(request.form.get("loan"))
        credit = safe_float(request.form.get("credit"))

        # Calculate risk score
        risk_score = DataProcessor.calculate_risk_score(age, income, loan_amount, credit)
        result = DataProcessor.get_risk_level(risk_score)

        # Save to database
        save_transaction_to_db(
            db, Transaction,
            user_id=session['user'],
            trans_type='risk',
            amount=loan_amount,
            location=request.form.get("location", "Unknown").strip(),
            device='risk_assessment',
            result=result,
            confidence=risk_score,
            details={'age': age, 'credit_score': credit, 'income': income}
        )

        flash(f"{result} - Score: {risk_score}/100", "info")
        print(f"⚠️ Risk Score: {result} ({risk_score}) - {session['user']}")

    except Exception as e:
        flash(f"❌ Calculation Error: {str(e)}", "danger")
        print(f"❌ Error in risk calculation: {e}")

    return redirect("/dashboard")

@app.route("/anomaly", methods=["POST"])
@login_required
def detect_anomaly():
    """Anomaly detection"""
    try:
        amount = safe_float(request.form.get("amount"))
        frequency = safe_int(request.form.get("frequency"))
        avg_amount = safe_float(request.form.get("avg_amount"))

        # Detect anomalies
        is_anomaly = False
        anomaly_reasons = []

        if frequency > 20:
            is_anomaly = True
            anomaly_reasons.append("High transaction frequency")

        if avg_amount > 0 and amount > avg_amount * 3:
            is_anomaly = True
            anomaly_reasons.append(f"Amount {amount / avg_amount:.1f}x higher than average")

        result = "🚨 Anomaly Detected" if is_anomaly else "✅ Normal Pattern"
        confidence = 85.0 if is_anomaly else 15.0

        # Save to database
        save_transaction_to_db(
            db, Transaction,
            user_id=session['user'],
            trans_type='anomaly',
            amount=amount,
            location=request.form.get("location", "Unknown").strip(),
            device='anomaly_detection',
            result=result,
            confidence=confidence,
            details={'reasons': anomaly_reasons, 'frequency': frequency}
        )

        flash(f"{result} - {' | '.join(anomaly_reasons) if anomaly_reasons else 'No anomalies'}", "warning" if is_anomaly else "success")
        print(f"🚨 Anomaly: {result} - {session['user']}")

    except Exception as e:
        flash(f"❌ Detection Error: {str(e)}", "danger")
        print(f"❌ Error in anomaly detection: {e}")

    return redirect("/dashboard")

@app.route("/spending", methods=["POST"])
@login_required
def analyze_spending():
    """Spending pattern analysis"""
    try:
        amount = safe_float(request.form.get("amount"))
        frequency = safe_int(request.form.get("frequency"))

        avg_trans = amount / frequency if frequency > 0 else 0

        # Categorize spending
        patterns = {
            100: "High-Frequency Spender",
            50: "Regular Spender",
            20: "Moderate Spender",
            5: "Occasional Spender",
            0: "Rare Spender"
        }
        
        pattern = "Rare Spender"
        for threshold, name in sorted(patterns.items(), reverse=True):
            if frequency > threshold:
                pattern = name
                break

        # Save to database
        save_transaction_to_db(
            db, Transaction,
            user_id=session['user'],
            trans_type='spending',
            amount=amount,
            location=request.form.get("location", "Unknown").strip(),
            device='spending_analysis',
            result=f"💰 {pattern}",
            confidence=75.0,
            details={'frequency': frequency, 'avg_transaction': avg_trans}
        )

        flash(f"Pattern: {pattern} | Avg: ${avg_trans:.2f}", "info")
        print(f"💰 Spending: {pattern} - {session['user']}")

    except Exception as e:
        flash(f"❌ Analysis Error: {str(e)}", "danger")
        print(f"❌ Error in spending analysis: {e}")

    return redirect("/dashboard")

# ============ CSV BULK UPLOAD ============
@app.route("/bulk-upload", methods=["POST"])
@login_required
def bulk_upload():
    """Process bulk CSV file upload"""
    try:
        if 'csv_file' not in request.files:
            flash("❌ No file selected", "danger")
            return redirect("/dashboard")

        file = request.files['csv_file']
        # FIX: Check if file AND filename exist before calling endswith
        if not file or not file.filename or file.filename == '' or not file.filename.endswith('.csv'):
            flash("❌ Only CSV files allowed", "danger")
            return redirect("/dashboard")

        # Process CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(stream)

        processed = 0
        errors = 0

        for row in csv_reader:
            try:
                # Safety check: Models?
                if not MODELS or MODELS['cc_rf'] is None:
                    flash("❌ Models not loaded", "danger")
                    return redirect("/dashboard")
                
                amount = safe_float(row.get('Amount', 0))
                hour = safe_int(row.get('Time', 12))
                device = row.get('Device', 'mobile').strip()
                location = row.get('Location', 'Unknown').strip()

                # Safety check: Metadata?
                if "feature_names" not in MODELS['cc_meta']:
                    raise Exception("Model metadata not found")

                # Prepare features
                from utils import calculate_fraud_features
                features = calculate_fraud_features(amount, hour, device, location)
                df = pd.DataFrame([features], columns=MODELS['cc_meta']["feature_names"])

                # Predict
                pred, prob = ml_manager.predict_fraud(MODELS['cc_rf'], features)
                confidence = round(prob, 2) if prob is not None else 0.0
                result = "⚠️ Fraud Detected" if pred == 1 else "✅ Safe"

                # Save to database
                save_transaction_to_db(
                    db, Transaction,
                    user_id=session['user'],
                    trans_type='fraud',
                    amount=amount,
                    location=location,
                    device=device,
                    result=result,
                    confidence=confidence,
                    details={'bulk_upload': True, 'hour': hour}
                )

                processed += 1
            except Exception as e:
                errors += 1
                print(f"Error processing CSV row: {e}")

        flash(f"✅ Processed {processed} records | ⚠️ {errors} errors", "success")
        print(f"📤 CSV Upload: {processed} processed, {errors} errors - {session['user']}")

    except Exception as e:
        flash(f"❌ Error processing file: {str(e)}", "danger")
        print(f"❌ Error in bulk upload: {e}")

    return redirect("/dashboard")

# ============ PDF REPORT GENERATION ============
@app.route("/download-report", methods=["GET"])
@login_required
def download_report():
    """Generate and download PDF report"""
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle  # type: ignore
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.lib import colors  # type: ignore

        # Get transactions
        transactions = Transaction.query.filter_by(user_id=session['user']).order_by(
            Transaction.date.desc()
        ).limit(20).all()

        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1
        )

        # Title
        elements.append(Paragraph("📊 Financial Fraud Detection Report", title_style))
        elements.append(Spacer(1, 0.2*inch))

        # Info
        info_text = f"""
        <b>User:</b> {session['user']} | <b>Role:</b> {session.get('role', 'Analyst')} | 
        <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        elements.append(Paragraph(info_text, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))

        # Statistics
        fraud_count = len([t for t in transactions if 'Fraud' in t.result])
        safe_count = len([t for t in transactions if 'Safe' in t.result])

        if len(transactions) > 0:
            detection_rate = (fraud_count / len(transactions) * 100)
        else:
            detection_rate = 0

        stats_text = f"""
        <b>📈 Statistics:</b><br/>
        Total Transactions: {len(transactions)}<br/>
        Fraud Detected: {fraud_count}<br/>
        Safe Transactions: {safe_count}<br/>
        Detection Rate: {detection_rate:.1f}%
        """
        elements.append(Paragraph(stats_text, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))

        # Table
        if transactions:
            data = [['Date', 'Type', 'Amount', 'Result', 'Confidence']]
            for trans in transactions:
                data.append([
                    trans.date.strftime('%Y-%m-%d'),
                    trans.trans_type.upper(),
                    f"${trans.amount:.2f}",
                    trans.result[:20],
                    f"{trans.confidence:.1f}%"
                ])

            table = Table(data, colWidths=[1.2*inch, 1*inch, 1*inch, 1.8*inch, 1.3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)

        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(
            "Report Generated by AI Financial Fraud Detection System",
            styles['Normal']
        ))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)

        print(f"📄 Report Generated - {session['user']}")
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'fraud_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

    except Exception as e:
        flash(f"❌ Error generating report: {str(e)}", "danger")
        print(f"❌ Error in report generation: {e}")
        return redirect("/dashboard")

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template("500.html"), 500

@app.errorhandler(403)
def forbidden(error):
    """Handle 403 errors"""
    return render_template("404.html"), 403

# ============ APPLICATION INITIALIZATION ============
def create_database():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("=" * 60)
    print("🏦 AI FINANCIAL FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 Model Directory: {MODEL_DIR}")
    print(f"🗄️  Database: fraud_detection.db")
    print("=" * 60)
    
    # Create database
    create_database()
    
    # Start server
    print("✅ Server starting...")
    print("🌐 Access at: http://127.0.0.1:5000")
    print("📝 Demo Credentials: admin / demo123")
    print("=" * 60)
    
    app.run(debug=True, host="127.0.0.1", port=5000)
