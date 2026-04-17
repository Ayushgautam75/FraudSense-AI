"""
Train all Project 3 models: CC fraud (RF, LR), loan default, risk (same as loan),
transaction anomaly (Isolation Forest), spending clusters (KMeans).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "model" / "project3"
CC_PATH = BASE_DIR / "creditcard.csv"
LOAN_PATH = BASE_DIR / "loan_data.csv"
BANK_PATH = BASE_DIR / "bank_transactions_data_2.csv"


def ensure_data():
    missing = [p for p in (CC_PATH, LOAN_PATH, BANK_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing: " + ", ".join(str(p) for p in missing))


def cc_features_from_row(amount: float, hour: int, device: str, location: str) -> list[float]:
    is_night = 1 if hour >= 20 or hour <= 6 else 0
    is_high = 1 if amount > 10000 else 0
    new_device = 1 if device.lower() != "mobile" else 0
    location_change = 1 if location.strip().lower() != "delhi" else 0
    return [amount, is_night, is_high, new_device, location_change]


def train_credit_fraud(df_cc: pd.DataFrame) -> None:
    df = df_cc.copy()
    hour = ((df["Time"] // 3600) % 24).astype(int)
    df["is_night"] = hour.apply(lambda h: 1 if h >= 20 or h <= 6 else 0)
    df["is_high"] = (df["Amount"] > 10000).astype(int)
    df["new_device"] = 0
    df["location_change"] = 0
    feat_cols = ["Amount", "is_night", "is_high", "new_device", "location_change"]
    X = df[feat_cols].astype(float)
    y = df["Class"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    joblib.dump(rf, OUT_DIR / "cc_fraud_rf.pkl")

    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    lr_pipe.fit(X, y)
    joblib.dump(lr_pipe, OUT_DIR / "cc_fraud_lr.pkl")

    meta = {
        "feature_names": feat_cols,
        "models": ["Random Forest", "Logistic Regression"],
    }
    with open(OUT_DIR / "cc_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_loan(df_loan: pd.DataFrame) -> None:
    df = df_loan.dropna().copy()
    y = df["loan_status"].astype(int)

    num_cols = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
    ]
    cat_cols = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    ]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(df[num_cols + cat_cols], y)

    joblib.dump(pipe, OUT_DIR / "loan_default_rf.pkl")

    loan_meta = {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "target_note": "loan_status from dataset (0/1); higher probability for predicted class.",
    }
    with open(OUT_DIR / "loan_meta.json", "w", encoding="utf-8") as f:
        json.dump(loan_meta, f, indent=2)


def train_isolation_forest(df_bank: pd.DataFrame) -> None:
    df = df_bank.copy()
    df["Channel_enc"] = df["Channel"].astype(str).str.strip()
    ch_map = {c: i for i, c in enumerate(sorted(df["Channel_enc"].unique()))}
    df["channel_code"] = df["Channel_enc"].map(ch_map)

    num_cols = [
        "TransactionAmount",
        "AccountBalance",
        "CustomerAge",
        "TransactionDuration",
        "LoginAttempts",
        "channel_code",
    ]
    X = df[num_cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Xs)

    joblib.dump(scaler, OUT_DIR / "iso_scaler.pkl")
    joblib.dump(iso, OUT_DIR / "iso_forest.pkl")
    with open(OUT_DIR / "iso_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": num_cols,
                "channel_map": ch_map,
            },
            f,
            indent=2,
        )


def train_kmeans_spending(df_cc: pd.DataFrame) -> None:
    df = df_cc[["Time", "Amount"]].astype(float).copy()
    df["hour"] = ((df["Time"] // 3600) % 24).astype(float)
    sample = df.sample(n=min(80000, len(df)), random_state=42)
    sample = sample[["Amount", "hour"]]
    X = sample.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(Xs)
    labels = km.predict(Xs)
    sample = sample.reset_index(drop=True)
    centers_amount = []
    for k in range(4):
        mask = labels == k
        centers_amount.append(float(sample.loc[mask, "Amount"].mean()))

    order = np.argsort(centers_amount)
    tier_names = ["Budget / low activity", "Everyday spending", "Active / higher value", "Premium / large ticket"]
    cluster_display = {}
    for rank, cluster_idx in enumerate(order):
        cluster_display[str(cluster_idx)] = tier_names[rank]

    joblib.dump(scaler, OUT_DIR / "spend_scaler.pkl")
    joblib.dump(km, OUT_DIR / "spend_kmeans.pkl")
    with open(OUT_DIR / "spend_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": ["Amount", "hour"],
                "cluster_labels": cluster_display,
            },
            f,
            indent=2,
        )


def main() -> None:
    ensure_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading creditcard (sampled for speed)...")
    df_cc = pd.read_csv(CC_PATH)
    if len(df_cc) > 200000:
        df_cc, _ = train_test_split(
            df_cc,
            train_size=200000,
            stratify=df_cc["Class"],
            random_state=42,
        )

    print("Training credit-card fraud models...")
    train_credit_fraud(df_cc)

    print("Training loan default model...")
    df_loan = pd.read_csv(LOAN_PATH)
    train_loan(df_loan)

    print("Training transaction anomaly (Isolation Forest)...")
    df_bank = pd.read_csv(BANK_PATH)
    train_isolation_forest(df_bank)

    print("Training spending pattern clustering (KMeans)...")
    train_kmeans_spending(pd.read_csv(CC_PATH))

    print(f"Done. Artifacts saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
