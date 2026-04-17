#!/usr/bin/env python3
"""
Diagnostic Script - Verify All Fix Applied
============================================
Run this to check if all fixes are working properly.
"""

import sys
from pathlib import Path

print("\n" + "="*60)
print("🔍 DIAGNOSTIC TEST - Checking 3 Main Fixes")
print("="*60 + "\n")

# Test 1: Check if model files exist
print("✔️ TEST 1: Model Files Exist?")
print("-" * 40)
model_dir = Path("model/project3")
required_files = [
    "cc_fraud_rf.pkl",
    "cc_meta.json",
    "loan_default_rf.pkl",
    "loan_meta.json",
    "iso_forest.pkl",
    "iso_meta.json",
    "iso_scaler.pkl",
    "spend_kmeans.pkl",
    "spend_meta.json",
    "spend_scaler.pkl"
]

all_exist = True
for file in required_files:
    path = model_dir / file
    exists = "✅" if path.exists() else "❌"
    print(f"  {exists} {file}")
    if not path.exists():
        all_exist = False

if all_exist:
    print("\n✅ All model files found!")
else:
    print("\n❌ Missing model files! Check model/project3/ directory")
    sys.exit(1)

# Test 2: Check Python dependencies
print("\n\n✔️ TEST 2: Required Packages Installed?")
print("-" * 40)

required_packages = {
    'flask': 'Flask',
    'flask_sqlalchemy': 'Flask-SQLAlchemy',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'joblib': 'joblib',
    'reportlab': 'reportlab',
}

missing = []
for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✅ {package_name}")
    except ImportError:
        print(f"  ❌ {package_name} MISSING")
        missing.append(package_name)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print(f"   Run: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n✅ All packages installed!")

# Test 3: Check if app.py has safety checks
print("\n\n✔️ TEST 3: Safety Checks in app.py?")
print("-" * 40)

with open("app.py", "r") as f:
    content = f.read()

checks = {
    "Models None check": "MODELS['cc_rf'] is None",
    "CSV filename check": "not file or not file.filename",
    "Metadata check": "\"feature_names\" not in MODELS",
    "Error messages": "flash(\"❌ ML Models not loaded",
}

all_checks_present = True
for check_name, check_text in checks.items():
    present = check_text in content
    status = "✅" if present else "❌"
    print(f"  {status} {check_name}")
    if not present:
        all_checks_present = False

if all_checks_present:
    print("\n✅ All safety checks added!")
else:
    print("\n⚠️ Some checks might be missing")

# Final status
print("\n\n" + "="*60)
print("📊 DIAGNOSTIC COMPLETE")
print("="*60)

if all_exist and not missing and all_checks_present:
    print("\n✅ ALL TESTS PASSED - Ready to run app.py!")
    print("\n💡 Next step: python app.py")
else:
    print("\n❌ Some issues found. See above.")
    sys.exit(1)

print("\n" + "="*60 + "\n")
