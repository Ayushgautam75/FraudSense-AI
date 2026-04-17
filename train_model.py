import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CSV_PATH = BASE_DIR / "creditcard.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"

MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
if not CSV_PATH.exists():
    raise FileNotFoundError("creditcard.csv not found in the project directory")

df = pd.read_csv(CSV_PATH)

# New features
# Note: the dataset does not include real location/device columns,
# so these are placeholder fields used by the app.
df['is_night'] = df['Time'].apply(lambda x: 1 if x > 80000 else 0)
df['is_high'] = df['Amount'].apply(lambda x: 1 if x > 10000 else 0)
df['new_device'] = 0
df['location_change'] = 0

# Features & target
X = df[['Amount', 'is_night', 'is_high', 'new_device', 'location_change']]
y = df['Class']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, MODEL_PATH)

print("🔥 Model retrained successfully!")