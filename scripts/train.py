from pathlib import Path
import sys, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from app.features import FEATURES

rng=np.random.default_rng(42); n=5000
df=pd.DataFrame({
"age":rng.integers(21,70,n),
"income":rng.integers(18000,180000,n).astype(float),
"loan_amount":rng.integers(50000,1500000,n).astype(float),
"credit_score":np.clip(rng.normal(680,70,n),300,850),
"dpd":rng.choice([0,5,15,30,45,60,90,120,180],n),
"missed_payment_count":rng.integers(0,7,n),
"previous_ptp_count":rng.integers(0,7,n),
"collection_attempts":rng.integers(0,12,n),
"recent_payment_amount":rng.integers(0,30000,n).astype(float),
"days_since_last_payment":rng.integers(0,180,n)
})
df["loan_amount"]=df["loan_amount"].astype(float)
df["outstanding_amount"]=df.loan_amount*rng.uniform(.15,.9,n)
df["emi_amount"]=df.loan_amount*rng.uniform(.015,.055,n)
df["previous_ptp_kept_count"]=[rng.integers(0,x+1) for x in df.previous_ptp_count]
df["successful_contacts"]=[rng.integers(0,x+1) for x in df.collection_attempts]
df["ptp_kept_rate"]=df.previous_ptp_kept_count/np.maximum(df.previous_ptp_count,1)
df["contact_success_rate"]=df.successful_contacts/np.maximum(df.collection_attempts,1)
df["outstanding_to_income"]=df.outstanding_amount/np.maximum(df.income*12,1e-6)
df["emi_to_income"]=df.emi_amount/np.maximum(df.income,1e-6)
df["payment_to_outstanding"]=df.recent_payment_amount/np.maximum(df.outstanding_amount,1e-6)
df["dpd_bucket"]=(df.dpd//30).clip(upper=6)
df["behavior_stress_score"]=(
 .35*np.minimum(df.dpd/180,1)+.20*np.minimum(df.missed_payment_count/6,1)+
 .20*np.minimum(df.days_since_last_payment/90,1)+.15*(1-df.ptp_kept_rate)+
 .10*(1-df.contact_success_rate))
stress=df.behavior_stress_score.to_numpy()
pay=np.clip(.80-.55*stress+.0003*(df.credit_score-650)+.25*df.payment_to_outstanding,.02,.98)
ptp=np.clip(.20+.55*df.ptp_kept_rate+.20*df.contact_success_rate-.30*stress,.02,.98)
rec=np.clip(.65-.45*stress+.10*np.minimum(df.payment_to_outstanding,1),.02,.98)
df["will_pay"]=rng.binomial(1,pay)
df["ptp_kept"]=rng.binomial(1,ptp)
df["recovery_success"]=rng.binomial(1,rec)

(Path(ROOT/"models")).mkdir(exist_ok=True)
for target,name in [("will_pay","payment_model.joblib"),("ptp_kept","ptp_model.joblib"),("recovery_success","recovery_model.joblib")]:
    X=df[FEATURES]; y=df[target]
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    m=RandomForestClassifier(n_estimators=250,max_depth=10,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)
    m.fit(Xt,yt)
    print(target,"ROC-AUC",round(roc_auc_score(yv,m.predict_proba(Xv)[:,1]),4))
    joblib.dump(m,ROOT/"models"/name)
df.to_csv(ROOT/"data_sample.csv",index=False)
print("Training complete.")
