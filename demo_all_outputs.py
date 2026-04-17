"""
Run sample inputs through all Project 3 models and print results (terminal demo).
Usage: python demo_all_outputs.py
"""
from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Import after path is project root
import app as web

if not web.PROJECT3_READY:
    raise SystemExit("Project 3 models missing. Run: python train_project3_models.py")


def line(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    line("1) CARD PAYMENT CHECK (sample inputs)")
    amount, hour, location, device = 120.0, 14, "Mumbai", "mobile"
    print(f"  Input: amount={amount}, hour={hour}, city={location}, device={device}")
    vals = web.create_features(amount, hour, location, device)
    print(f"  Features: {dict(zip(web.cc_meta['feature_names'], vals))}")
    X = pd.DataFrame([vals], columns=web.cc_meta["feature_names"])
    pr_rf = int(web.cc_rf.predict(X)[0])
    pr_lr = int(web.cc_lr.predict(X)[0])
    proba_rf = web.cc_rf.predict_proba(X)[0]
    cls_rf = list(web.cc_rf.classes_)
    p_rf = float(proba_rf[cls_rf.index(1)]) if 1 in cls_rf else float(max(proba_rf))
    proba_lr = web.cc_lr.predict_proba(X)[0]
    cls_lr = list(web.cc_lr.classes_)
    p_lr = float(proba_lr[cls_lr.index(1)]) if 1 in cls_lr else float(max(proba_lr))
    print(f"  Screening A -> {'Review' if pr_rf == 1 else 'Clear'}  |  unusual score: {p_rf*100:.2f}%")
    print(f"  Screening B -> {'Review' if pr_lr == 1 else 'Clear'}  |  unusual score: {p_lr*100:.2f}%")
    print(f"  Both agree: {pr_rf == pr_lr}")

    line("2) LOAN APPLICATION (sample inputs)")
    row = {
        "person_age": 26.0,
        "person_income": 67048.0,
        "person_emp_exp": 4,
        "loan_amnt": 8000.0,
        "loan_int_rate": 11.01,
        "loan_percent_income": 0.12,
        "cb_person_cred_hist_length": 4.0,
        "credit_score": 640,
        "person_gender": "female",
        "person_education": "Bachelor",
        "person_home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "previous_loan_defaults_on_file": "No",
    }
    for k, v in row.items():
        print(f"  {k}: {v}")
    df = pd.DataFrame([row])
    pred = int(web.loan_pipe.predict(df)[0])
    proba = web.loan_pipe.predict_proba(df)[0]
    cls_loan = list(web.loan_pipe.classes_)
    p0 = float(proba[cls_loan.index(0)]) if 0 in cls_loan else float(max(proba))
    p1 = float(proba[cls_loan.index(1)]) if 1 in cls_loan else float(max(proba))
    print(f"  -> Predicted category: {pred}")
    print(f"  -> P(category 0): {p0*100:.2f}%   P(category 1): {p1*100:.2f}%")

    line("3) CUSTOMER RISK (same profile as loan)")
    risk_index = round(100 * p0, 1)
    print(f"  -> Risk index: {risk_index} / 100")
    print(f"  -> Band: {web._risk_band(risk_index)}")

    line("4) UNUSUAL ACTIVITY (sample inputs)")
    amt, bal, age, dur, logins = 180.0, 9500.0, 34.0, 72.0, 1.0
    channel = "Online"
    cmap = web.iso_meta["channel_map"]
    ch_code = float(cmap[channel])
    cols = web.iso_meta["feature_names"]
    vec = pd.DataFrame([[amt, bal, age, dur, logins, ch_code]], columns=cols)
    xs = web.iso_scaler.transform(vec)
    flag = int(web.iso_forest.predict(xs)[0])
    score = float(web.iso_forest.decision_function(xs)[0])
    print(f"  Input: amount={amt}, balance={bal}, age={age}, duration_sec={dur}, logins={logins}, channel={channel}")
    print(f"  -> {'Unusual (review)' if flag == -1 else 'Typical pattern'}")
    print(f"  -> Technical score: {score:.4f}")

    line("5) SPENDING STYLE (sample inputs)")
    amount_s, hour_s = 85.0, 15.0
    fn = web.spend_meta["feature_names"]
    Xs = pd.DataFrame([[amount_s, hour_s]], columns=fn)
    xs2 = web.spend_scaler.transform(Xs)
    cid = int(web.spend_km.predict(xs2)[0])
    name = web.spend_meta.get("cluster_labels", {}).get(str(cid), f"Cluster {cid}")
    print(f"  Input: amount={amount_s}, hour={hour_s}")
    print(f"  -> Segment: {name}  (id {cid})")

    line("DONE - same sample values work in the website forms")
    print()


if __name__ == "__main__":
    main()
