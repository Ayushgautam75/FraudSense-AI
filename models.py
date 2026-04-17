"""
Database models for the Financial Fraud Detection System
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Transaction(db.Model):
    """Transaction history model - stores all predictions"""
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False, index=True)
    trans_type = db.Column(db.String(50), nullable=False)  # fraud, loan, risk, anomaly, spending
    amount = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100))
    device = db.Column(db.String(50))
    result = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.now, index=True)
    details = db.Column(db.Text)  # JSON string with additional details

    def __repr__(self):
        return f'<Transaction {self.id} - {self.trans_type} - {self.result}>'

    def to_dict(self):
        """Convert transaction to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'trans_type': self.trans_type,
            'amount': self.amount,
            'location': self.location,
            'device': self.device,
            'result': self.result,
            'confidence': self.confidence,
            'date': self.date.strftime('%Y-%m-%d %H:%M:%S'),
            'details': json.loads(self.details) if self.details else {}
        }

    @staticmethod
    def get_user_statistics(user_id):
        """Get statistics for a specific user"""
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if not transactions:
            return {
                'total': 0,
                'fraud_detected': 0,
                'safe': 0,
                'avg_confidence': 0,
                'by_type': {}
            }
        
        fraud_count = len([t for t in transactions if 'Fraud' in t.result])
        safe_count = len([t for t in transactions if 'Safe' in t.result])
        avg_confidence = sum(t.confidence for t in transactions) / len(transactions)
        
        # Group by type
        by_type = {}
        for trans in transactions:
            if trans.trans_type not in by_type:
                by_type[trans.trans_type] = 0
            by_type[trans.trans_type] += 1
        
        return {
            'total': len(transactions),
            'fraud_detected': fraud_count,
            'safe': safe_count,
            'avg_confidence': round(avg_confidence, 2),
            'by_type': by_type
        }
