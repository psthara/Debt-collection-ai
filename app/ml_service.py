import joblib
import numpy as np
from .config import MODEL_DIR
from .features import make_features

class MLService:
    def __init__(self):
        self.payment = joblib.load(MODEL_DIR/"payment_model.joblib")
        self.ptp = joblib.load(MODEL_DIR/"ptp_model.joblib")
        self.recovery = joblib.load(MODEL_DIR/"recovery_model.joblib")

    def predict(self, customer):
        X = make_features(customer)
        p = float(self.payment.predict_proba(X)[0,1])
        t = float(self.ptp.predict_proba(X)[0,1])
        r = float(self.recovery.predict_proba(X)[0,1])
        c = customer.model_dump()
        exposure = min(c["outstanding_amount"]/max(c["income"]*12,1),2)/2
        score = 100*(.45*(1-p)+.25*(1-t)+.20*min(c["dpd"]/180,1)+.10*exposure)
        score = round(float(np.clip(score,0,100)),2)
        band = "HIGH" if score>=70 else "MEDIUM" if score>=40 else "LOW"
        return {
            "payment_probability_30d": round(p,4),
            "ptp_probability": round(t,4),
            "recovery_probability": round(r,4),
            "collection_priority_score": score,
            "priority_band": band
        }
