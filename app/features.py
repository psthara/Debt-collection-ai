import pandas as pd

FEATURES = [
"age","income","loan_amount","outstanding_amount","emi_amount","credit_score",
"dpd","missed_payment_count","previous_ptp_count","previous_ptp_kept_count",
"collection_attempts","successful_contacts","recent_payment_amount",
"days_since_last_payment","ptp_kept_rate","contact_success_rate",
"outstanding_to_income","emi_to_income","payment_to_outstanding",
"dpd_bucket","behavior_stress_score"
]

def make_features(obj):
    d = obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)
    eps = 1e-6
    d["ptp_kept_rate"] = d["previous_ptp_kept_count"] / max(d["previous_ptp_count"],1)
    d["contact_success_rate"] = d["successful_contacts"] / max(d["collection_attempts"],1)
    d["outstanding_to_income"] = d["outstanding_amount"] / max(d["income"]*12,eps)
    d["emi_to_income"] = d["emi_amount"] / max(d["income"],eps)
    d["payment_to_outstanding"] = d["recent_payment_amount"] / max(d["outstanding_amount"],eps)
    d["dpd_bucket"] = min(d["dpd"]//30,6)
    d["behavior_stress_score"] = (
        .35*min(d["dpd"]/180,1) +
        .20*min(d["missed_payment_count"]/6,1) +
        .20*min(d["days_since_last_payment"]/90,1) +
        .15*(1-d["ptp_kept_rate"]) +
        .10*(1-d["contact_success_rate"])
    )
    return pd.DataFrame([[d[x] for x in FEATURES]], columns=FEATURES)
