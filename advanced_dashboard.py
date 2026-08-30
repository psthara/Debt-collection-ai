import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime

# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"
MAX_CONTACTS_7_DAYS = 7          # e.g. US Reg F "7-in-7" style cap
QUIET_HOURS_START, QUIET_HOURS_END = 21, 8  # 9pm - 8am local, no outbound contact

FEATURES = [
    "age", "income", "loan_amount", "outstanding_amount", "emi_amount",
    "credit_score", "dpd", "missed_payment_count", "previous_ptp_count",
    "previous_ptp_kept_count", "collection_attempts", "successful_contacts",
    "recent_payment_amount", "days_since_last_payment",
]

CHANNELS = ["PHONE", "SMS", "WHATSAPP", "EMAIL"]

st.set_page_config(
    page_title="AI Debt Collection Intelligence — Advanced",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# SYNTHETIC DATA (with a real historical outcome + treatment log,
# so models are TRAINED rather than hand-coded formulas)
# ============================================================

@st.cache_data
def load_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "customer_id": [f"C{1000+i}" for i in range(n)],
        "age": rng.integers(21, 70, n),
        "income": rng.integers(20000, 150000, n),
        "loan_amount": rng.integers(50000, 1500000, n),
        "outstanding_amount": rng.integers(10000, 800000, n),
        "emi_amount": rng.integers(2000, 50000, n),
        "credit_score": rng.integers(450, 850, n),
        "dpd": rng.choice([0, 5, 15, 30, 45, 60, 90, 120, 180], n),
        "missed_payment_count": rng.integers(0, 7, n),
        "previous_ptp_count": rng.integers(0, 7, n),
        "previous_ptp_kept_count": rng.integers(0, 6, n),
        "collection_attempts": rng.integers(0, 12, n),
        "successful_contacts": rng.integers(0, 8, n),
        "recent_payment_amount": rng.integers(0, 30000, n),
        "days_since_last_payment": rng.integers(0, 180, n),
        "preferred_channel": rng.choice(CHANNELS, n),
        "contacts_last_7_days": rng.integers(0, 10, n),
    })
    df["previous_ptp_kept_count"] = np.minimum(df["previous_ptp_kept_count"], df["previous_ptp_count"])

    # Latent "financial stress" driving true (unobserved) repayment behavior.
    # Models below are TRAINED on outcomes generated from this - they don't
    # see this formula, they have to learn it from data like a real model would.
    stress = (
        0.40 * np.minimum(df["dpd"] / 180, 1)
        + 0.20 * np.minimum(df["missed_payment_count"] / 6, 1)
        + 0.20 * np.minimum(df["days_since_last_payment"] / 180, 1)
        + 0.20 * (1 - df["previous_ptp_kept_count"] / np.maximum(df["previous_ptp_count"], 1))
    )

    # Randomized historical treatment (contacted vs not) -> lets us do
    # unbiased uplift estimation (like an RCT / historical A-B log).
    df["treatment"] = rng.binomial(1, 0.5, n)

    # True lift from being contacted is heterogeneous: mid-stress customers
    # respond most; very low/high stress customers barely move.
    responsiveness = np.exp(-((stress - 0.45) ** 2) / 0.06)
    true_uplift = 0.18 * responsiveness

    base_pay_prob = np.clip(0.90 - stress + rng.normal(0, 0.05, n), 0.02, 0.97)
    pay_prob = np.clip(base_pay_prob + df["treatment"] * true_uplift, 0.02, 0.99)
    df["did_pay_last_cycle"] = rng.binomial(1, pay_prob)

    # kept a promise-to-pay outcome (separate label, used for PTP model)
    ptp_base = np.clip(0.85 - stress * 0.9 + rng.normal(0, 0.05, n), 0.02, 0.97)
    df["kept_ptp_last_cycle"] = rng.binomial(1, np.clip(ptp_base + df["treatment"] * true_uplift * 0.7, 0.02, 0.99))

    df["_stress_debug"] = stress  # kept only for internal sanity checks, not shown
    return df


df_raw = load_data()

# ============================================================
# TRAIN REAL MODELS (cached as resources so training happens once)
# ============================================================

@st.cache_resource
def train_models(df):
    X = df[FEATURES]

    # --- 1. Payment probability model (trained + validated, not a formula) ---
    y_pay = df["did_pay_last_cycle"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_pay, test_size=0.25, random_state=42, stratify=y_pay
    )
    pay_model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    pay_model.fit(X_train, y_train)
    pay_auc = roc_auc_score(y_test, pay_model.predict_proba(X_test)[:, 1])

    # --- 2. PTP-kept model ---
    y_ptp = df["kept_ptp_last_cycle"]
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X, y_ptp, test_size=0.25, random_state=42, stratify=y_ptp
    )
    ptp_model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    ptp_model.fit(Xp_train, yp_train)
    ptp_auc = roc_auc_score(yp_test, ptp_model.predict_proba(Xp_test)[:, 1])

    # --- 3. Uplift model (T-learner): separate models for treated vs control,
    #         uplift = P(pay | treated, x) - P(pay | control, x). Because
    #         treatment was randomized in the synthetic history, this is an
    #         unbiased individual treatment effect estimate. ---
    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    model_treated = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model_control = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model_treated.fit(treated[FEATURES], treated["did_pay_last_cycle"])
    model_control.fit(control[FEATURES], control["did_pay_last_cycle"])

    # --- 4. Feature importance via permutation (model-agnostic, honest) ---
    perm = permutation_importance(pay_model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    return {
        "pay_model": pay_model, "pay_auc": pay_auc,
        "ptp_model": ptp_model, "ptp_auc": ptp_auc,
        "model_treated": model_treated, "model_control": model_control,
        "importance_df": importance_df,
    }


models = train_models(df_raw)

# ============================================================
# SCORE THE FULL PORTFOLIO WITH TRAINED MODELS
# ============================================================

@st.cache_data
def score_portfolio(df, _models):
    df = df.copy()
    X = df[FEATURES]

    df["payment_probability"] = _models["pay_model"].predict_proba(X)[:, 1]
    df["ptp_probability"] = _models["ptp_model"].predict_proba(X)[:, 1]

    p_treated = _models["model_treated"].predict_proba(X)[:, 1]
    p_control = _models["model_control"].predict_proba(X)[:, 1]
    df["uplift_score"] = p_treated - p_control          # expected lift FROM contacting them
    df["recovery_probability"] = p_treated               # probability if contacted

    # --- Segmentation via unsupervised clustering (KMeans) ---
    scaler = StandardScaler()
    cluster_features = ["dpd", "missed_payment_count", "days_since_last_payment",
                         "outstanding_amount", "uplift_score", "payment_probability"]
    X_scaled = scaler.fit_transform(df[cluster_features])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["segment_raw"] = km.fit_predict(X_scaled)

    # Label clusters by their characteristics (highest uplift, self-cure, etc.)
    seg_summary = df.groupby("segment_raw")[["uplift_score", "payment_probability", "dpd"]].mean()
    seg_summary = seg_summary.sort_values("uplift_score", ascending=False)
    ordered_ids = seg_summary.index.tolist()
    labels = ["High-Value Responders", "Steady Self-Payers", "Needs Restructuring", "Hard-to-Recover"]
    # Reorder by a sensible business heuristic
    seg_summary2 = df.groupby("segment_raw")["payment_probability"].mean().sort_values(ascending=False)
    self_payer_id = seg_summary2.index[0]
    label_map = {}
    remaining_labels = labels.copy()
    label_map[self_payer_id] = "Steady Self-Payers"
    remaining_labels.remove("Steady Self-Payers")
    remaining_ids = [i for i in ordered_ids if i != self_payer_id]
    uplift_sorted = df[df["segment_raw"].isin(remaining_ids)].groupby("segment_raw")["uplift_score"].mean().sort_values(ascending=False)
    for i, seg_id in enumerate(uplift_sorted.index):
        label_map[seg_id] = remaining_labels[min(i, len(remaining_labels) - 1)]

    df["segment"] = df["segment_raw"].map(label_map)

    # --- Priority score: risk-adjusted AND uplift-adjusted, not raw risk alone ---
    # Rewards contacting people who are likely to respond TO contact, not just
    # people who look risky (some risky accounts self-cure regardless).
    df["priority_score"] = np.clip(
        (1 - df["payment_probability"]) * 30
        + df["uplift_score"].clip(lower=0) * 100 * 0.35
        + np.minimum(df["dpd"] / 180, 1) * 20
        + np.minimum(df["outstanding_amount"] / np.maximum(df["income"] * 12, 1), 1) * 15,
        0, 100
    )

    def priority_bucket(x):
        if x >= 65:
            return "HIGH"
        elif x >= 38:
            return "MEDIUM"
        return "LOW"

    df["priority"] = df["priority_score"].apply(priority_bucket)

    # --- Compliance flags ---
    df["compliance_flag"] = np.where(
        df["contacts_last_7_days"] >= MAX_CONTACTS_7_DAYS, "OVER LIMIT", "OK"
    )

    return df


df = score_portfolio(df_raw, models)

# ============================================================
# BANDIT STATE FOR CHANNEL SELECTION (Thompson Sampling)
# persists across reruns via session_state -> genuinely "learns"
# as you click "simulate a day of outcomes"
# ============================================================

def init_bandit_state():
    if "bandit_alpha" not in st.session_state:
        segments = df["segment"].unique().tolist()
        st.session_state.bandit_alpha = {seg: {c: 1.0 for c in CHANNELS} for seg in segments}
        st.session_state.bandit_beta = {seg: {c: 1.0 for c in CHANNELS} for seg in segments}
        st.session_state.bandit_rounds = 0


def thompson_pick(segment):
    a = st.session_state.bandit_alpha[segment]
    b = st.session_state.bandit_beta[segment]
    samples = {c: np.random.beta(a[c], b[c]) for c in CHANNELS}
    return max(samples, key=samples.get), samples


def simulate_bandit_round(n_customers=40):
    # Simulate a batch of contact outcomes and update posteriors.
    # True (hidden) channel effectiveness per segment - the bandit doesn't
    # know this, it has to discover it through simulated trials.
    true_rates = {
        "High-Value Responders": {"PHONE": 0.55, "SMS": 0.35, "WHATSAPP": 0.50, "EMAIL": 0.20},
        "Steady Self-Payers": {"PHONE": 0.30, "SMS": 0.45, "WHATSAPP": 0.40, "EMAIL": 0.35},
        "Needs Restructuring": {"PHONE": 0.45, "SMS": 0.25, "WHATSAPP": 0.42, "EMAIL": 0.18},
        "Hard-to-Recover": {"PHONE": 0.15, "SMS": 0.12, "WHATSAPP": 0.14, "EMAIL": 0.08},
    }
    sample = df.sample(min(n_customers, len(df)), random_state=None)
    for _, row in sample.iterrows():
        seg = row["segment"]
        channel, _ = thompson_pick(seg)
        p_success = true_rates.get(seg, {}).get(channel, 0.2)
        outcome = np.random.binomial(1, p_success)
        if outcome:
            st.session_state.bandit_alpha[seg][channel] += 1
        else:
            st.session_state.bandit_beta[seg][channel] += 1
    st.session_state.bandit_rounds += 1


init_bandit_state()

# ============================================================
# HEADER
# ============================================================

st.title("💳 AI Debt Collection Intelligence — Advanced")
st.caption(
    "Trained ML models (not hardcoded rules) · Uplift-based prioritization · "
    "Thompson-sampling bandit for channel selection · Segmentation · Compliance guardrails · RAG + GenAI"
)
st.divider()

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔎 Filters")

priority_filter = st.sidebar.multiselect(
    "Priority", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
)
segment_filter = st.sidebar.multiselect(
    "Segment", options=sorted(df["segment"].unique()), default=sorted(df["segment"].unique())
)
dpd_filter = st.sidebar.slider("Minimum DPD", 0, 180, 0)
channel_filter = st.sidebar.multiselect(
    "Preferred Channel", options=df["preferred_channel"].unique().tolist(),
    default=df["preferred_channel"].unique().tolist()
)
st.sidebar.divider()
in_quiet_hours = st.sidebar.checkbox("Simulate: currently in quiet hours (9pm–8am)", value=False)
st.sidebar.caption(f"Compliance cap: max {MAX_CONTACTS_7_DAYS} contacts / 7 days")

filtered_df = df[
    (df["priority"].isin(priority_filter))
    & (df["segment"].isin(segment_filter))
    & (df["dpd"] >= dpd_filter)
    & (df["preferred_channel"].isin(channel_filter))
]

# ============================================================
# KPI ROW
# ============================================================

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Customers", f"{len(filtered_df):,}")
col2.metric("Outstanding", f"₹{filtered_df['outstanding_amount'].sum()/1e7:.2f} Cr")
col3.metric("High Risk", f"{(filtered_df['priority']=='HIGH').sum():,}")
col4.metric("Avg Payment Prob.", f"{filtered_df['payment_probability'].mean()*100:.1f}%")
col5.metric("Avg Uplift from Contact", f"{filtered_df['uplift_score'].mean()*100:+.1f} pts")
col6.metric("Compliance Violations", f"{(filtered_df['compliance_flag']=='OVER LIMIT').sum():,}")

st.divider()

# ============================================================
# TABS
# ============================================================

tab_overview, tab_model, tab_nba, tab_segments, tab_customer, tab_compliance, tab_rag, tab_genai = st.tabs(
    ["📊 Overview", "🧠 Model Performance", "🎯 Bandit NBA", "🧩 Segments",
     "👤 Customer 360", "🛡️ Compliance", "📚 Policy Assistant", "✍️ GenAI Message"]
)

# ---- OVERVIEW ----
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered_df, x="priority_score", color="priority", nbins=30,
                            title="Priority Score Distribution",
                            color_discrete_map={"HIGH": "#d62728", "MEDIUM": "#ff9f1c", "LOW": "#2ca02c"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(filtered_df, x="payment_probability", y="uplift_score",
                          color="segment", size="outstanding_amount", hover_data=["customer_id", "dpd"],
                          title="Payment Probability vs. Uplift from Contact",
                          labels={"payment_probability": "P(will pay)", "uplift_score": "Uplift from contact"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "> **Reading this chart:** bottom-right (high self-pay probability, low uplift) customers "
        "don't need aggressive contact — they'll likely pay anyway. Top-middle (moderate pay probability, "
        "high uplift) is where collection effort has the biggest marginal impact."
    )

    st.subheader("🚨 Priority Queue (ranked by uplift-adjusted priority score)")
    display_cols = ["customer_id", "segment", "dpd", "outstanding_amount", "payment_probability",
                     "uplift_score", "priority_score", "priority", "compliance_flag", "preferred_channel"]
    disp = filtered_df[display_cols].sort_values("priority_score", ascending=False).head(50).copy()
    disp["payment_probability"] *= 100
    disp["uplift_score"] *= 100
    st.dataframe(
        disp.style.format({
            "outstanding_amount": "₹{:,.0f}",
            "payment_probability": "{:.1f}%",
            "uplift_score": "{:+.1f} pts",
            "priority_score": "{:.1f}",
        }).map(lambda v: "background-color:#3a1c1c" if v == "OVER LIMIT" else "", subset=["compliance_flag"]),
        use_container_width=True, hide_index=True
    )

# ---- MODEL PERFORMANCE ----
with tab_model:
    st.subheader("Model validation (held-out test set)")
    c1, c2 = st.columns(2)
    c1.metric("Payment model AUC", f"{models['pay_auc']:.3f}")
    c2.metric("PTP-kept model AUC", f"{models['ptp_auc']:.3f}")
    st.caption(
        "AUC = 0.5 is random guessing, 1.0 is perfect separation. These are GradientBoosting "
        "classifiers trained on historical outcomes and evaluated on a held-out 25% test split — "
        "the scores above are learned from data, not hand-set weights."
    )

    st.subheader("What's driving the payment-probability model (permutation importance)")
    fig = px.bar(models["importance_df"], x="importance", y="feature", orientation="h",
                 title="Feature importance (drop in model accuracy when feature is shuffled)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Uplift model (T-learner) sanity check")
    st.caption(
        "Two separate models trained on historically-contacted vs. not-contacted customers "
        "(treatment was randomized historically, so this is an unbiased effect estimate). "
        "The distribution below should show real heterogeneity — some customers respond strongly "
        "to contact, others barely move."
    )
    fig = px.histogram(df, x="uplift_score", nbins=40, title="Distribution of estimated individual uplift")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# ---- BANDIT NBA ----
with tab_nba:
    st.subheader("🎯 Contextual bandit for channel selection (Thompson Sampling)")
    st.caption(
        "Instead of a fixed if/else rule, each segment maintains a Beta(α, β) posterior per channel. "
        "Click below to simulate a batch of real-world contact outcomes — watch the recommended "
        "channel per segment shift as the bandit learns which channel actually converts."
    )

    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("▶️ Simulate a batch of contact outcomes"):
            simulate_bandit_round()
        st.metric("Rounds simulated", st.session_state.bandit_rounds)
        if st.button("🔄 Reset bandit learning"):
            for key in ["bandit_alpha", "bandit_beta", "bandit_rounds"]:
                if key in st.session_state:
                    del st.session_state[key]
            init_bandit_state()
            st.rerun()

    with b2:
        rows = []
        for seg in st.session_state.bandit_alpha:
            for ch in CHANNELS:
                a = st.session_state.bandit_alpha[seg][ch]
                b = st.session_state.bandit_beta[seg][ch]
                est_rate = a / (a + b)
                rows.append({"segment": seg, "channel": ch, "estimated_success_rate": est_rate,
                             "trials": int(a + b - 2)})
        bandit_df = pd.DataFrame(rows)
        fig = px.bar(bandit_df, x="segment", y="estimated_success_rate", color="channel",
                     barmode="group", title="Learned channel effectiveness by segment",
                     labels={"estimated_success_rate": "Estimated success rate"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current recommended channel per segment (posterior mean)")
    best_per_seg = bandit_df.loc[bandit_df.groupby("segment")["estimated_success_rate"].idxmax()]
    st.dataframe(
        best_per_seg[["segment", "channel", "estimated_success_rate", "trials"]]
        .rename(columns={"channel": "recommended_channel"})
        .style.format({"estimated_success_rate": "{:.1%}"}),
        use_container_width=True, hide_index=True
    )

# ---- SEGMENTS ----
with tab_segments:
    st.subheader("🧩 Unsupervised customer segments (KMeans on behavioral features)")
    seg_stats = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_dpd=("dpd", "mean"),
        avg_outstanding=("outstanding_amount", "mean"),
        avg_payment_prob=("payment_probability", "mean"),
        avg_uplift=("uplift_score", "mean"),
    ).reset_index().sort_values("avg_uplift", ascending=False)
    st.dataframe(
        seg_stats.style.format({
            "avg_dpd": "{:.0f}", "avg_outstanding": "₹{:,.0f}",
            "avg_payment_prob": "{:.1%}", "avg_uplift": "{:+.1%}"
        }), use_container_width=True, hide_index=True
    )

    fig = px.sunburst(df, path=["segment", "priority"], values="outstanding_amount",
                       title="Outstanding amount by segment → priority")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- **High-Value Responders** — moderate risk, high uplift → prioritize live contact.\n"
        "- **Steady Self-Payers** — will likely pay regardless → light-touch reminders only, save capacity.\n"
        "- **Needs Restructuring** — high risk but still responsive → offer payment plans, not pressure.\n"
        "- **Hard-to-Recover** — low probability, low uplift → route to write-off review / legal, not repeated outreach."
    )

# ---- CUSTOMER 360 ----
with tab_customer:
    customer_id = st.selectbox("Select Customer", filtered_df["customer_id"].tolist())
    customer = filtered_df[filtered_df["customer_id"] == customer_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("### Customer Information")
        st.write(f"**Customer ID:** {customer['customer_id']}")
        st.write(f"**Segment:** {customer['segment']}")
        st.write(f"**Age:** {customer['age']}")
        st.write(f"**Income:** ₹{customer['income']:,.0f}")
        st.write(f"**Credit Score:** {customer['credit_score']}")
    with c2:
        st.write("### Loan Information")
        st.write(f"**Loan Amount:** ₹{customer['loan_amount']:,.0f}")
        st.write(f"**Outstanding:** ₹{customer['outstanding_amount']:,.0f}")
        st.write(f"**EMI:** ₹{customer['emi_amount']:,.0f}")
        st.write(f"**DPD:** {customer['dpd']}")
        st.write(f"**Contacts (7d):** {customer['contacts_last_7_days']}")
    with c3:
        st.write("### AI Insights")
        st.metric("Payment Probability", f"{customer['payment_probability']*100:.1f}%")
        st.metric("PTP Probability", f"{customer['ptp_probability']*100:.1f}%")
        st.metric("Uplift from Contact", f"{customer['uplift_score']*100:+.1f} pts")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=customer["priority_score"],
        title={"text": "Priority Score"},
        gauge={"axis": {"range": [0, 100]},
               "steps": [{"range": [0, 38], "color": "#1e3d1e"},
                         {"range": [38, 65], "color": "#4d3a10"},
                         {"range": [65, 100], "color": "#4d1414"}],
               "bar": {"color": "white"}}
    ))
    gauge.update_layout(height=250, margin=dict(t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("🎯 Next Best Action")
    recommended_channel, posterior_samples = thompson_pick(customer["segment"])

    blocked_reason = None
    if customer["compliance_flag"] == "OVER LIMIT":
        blocked_reason = f"Contact cap reached ({customer['contacts_last_7_days']}/{MAX_CONTACTS_7_DAYS} in 7 days)"
    elif in_quiet_hours:
        blocked_reason = "Outside permitted contact hours"

    if blocked_reason:
        st.error(f"🚫 Action blocked by compliance guardrail: {blocked_reason}")
    else:
        if customer["dpd"] >= 90:
            action = "Escalate to Human Review"
        elif customer["uplift_score"] < 0.02:
            action = "Light-touch reminder only (low marginal value of contact)"
        elif customer["ptp_probability"] >= 0.6:
            action = "Request Promise-to-Pay"
        else:
            action = "Repayment / Restructuring Discussion"

        st.success(
            f"**Action:** {action}  \n"
            f"**Channel (bandit-recommended):** {recommended_channel}  \n"
            f"**Segment:** {customer['segment']}  \n"
            f"**Reason:** DPD={customer['dpd']}d, PayProb={customer['payment_probability']*100:.0f}%, "
            f"PTPProb={customer['ptp_probability']*100:.0f}%, Uplift={customer['uplift_score']*100:+.1f}pts"
        )

# ---- COMPLIANCE ----
with tab_compliance:
    st.subheader("🛡️ Compliance guardrails")
    st.caption(
        "These checks run BEFORE any recommended action is allowed to execute — "
        "the model can recommend contact, but this layer can veto it."
    )
    violations = df[df["compliance_flag"] == "OVER LIMIT"]
    c1, c2 = st.columns(2)
    c1.metric("Accounts over contact-frequency cap", len(violations))
    c2.metric("% of portfolio", f"{len(violations)/len(df)*100:.1f}%")

    fig = px.histogram(df, x="contacts_last_7_days", nbins=11,
                        title="Contacts in last 7 days across portfolio")
    fig.add_vline(x=MAX_CONTACTS_7_DAYS, line_dash="dash", line_color="red",
                  annotation_text="cap")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        violations[["customer_id", "segment", "contacts_last_7_days", "priority", "preferred_channel"]]
        .sort_values("contacts_last_7_days", ascending=False),
        use_container_width=True, hide_index=True
    )
    st.info(
        "In production this layer should also encode: quiet-hour windows by timezone, "
        "do-not-contact / dispute flags, cease-and-desist letters, and jurisdiction-specific rules — "
        "and it should be enforced server-side, not just in the dashboard."
    )

# ---- RAG ASSISTANT ----
with tab_rag:
    st.subheader("📚 AI Collection Policy Assistant")
    question = st.text_input("Ask a question about collection policy")
    if st.button("Ask AI Policy Assistant"):
        if question.strip():
            try:
                response = requests.post(f"{API_URL}/rag/query", json={"question": question}, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    st.success(result.get("answer", "No answer"))
                    for source in result.get("sources", []):
                        st.write(f"📄 {source.get('title')}")
                else:
                    st.error(response.text)
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend: {e}")

# ---- GENAI MESSAGE ----
with tab_genai:
    st.subheader("✍️ GenAI Collection Message")
    if "customer_id" not in dir():
        st.info("Select a customer in the Customer 360 tab first.")
    else:
        tone = st.selectbox("Message Tone", ["professional", "empathetic", "concise"])
        if st.button("Generate Collection Message"):
            payload = {
                "customer": {
                    "customer_id": customer["customer_id"], "age": int(customer["age"]),
                    "income": float(customer["income"]), "loan_amount": float(customer["loan_amount"]),
                    "outstanding_amount": float(customer["outstanding_amount"]),
                    "emi_amount": float(customer["emi_amount"]), "credit_score": float(customer["credit_score"]),
                    "dpd": int(customer["dpd"]), "missed_payment_count": int(customer["missed_payment_count"]),
                    "previous_ptp_count": int(customer["previous_ptp_count"]),
                    "previous_ptp_kept_count": int(customer["previous_ptp_kept_count"]),
                    "collection_attempts": int(customer["collection_attempts"]),
                    "successful_contacts": int(customer["successful_contacts"]),
                    "recent_payment_amount": float(customer["recent_payment_amount"]),
                    "days_since_last_payment": int(customer["days_since_last_payment"]),
                    "preferred_channel": customer["preferred_channel"],
                },
                "tone": tone,
            }
            try:
                response = requests.post(f"{API_URL}/generate-message", json=payload, timeout=60)
                if response.status_code == 200:
                    st.info(response.json().get("message", ""))
                else:
                    st.error(response.text)
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend: {e}")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "AI Debt Collection Intelligence Platform (Advanced) | "
    "Models are trained on historical outcomes and validated on held-out data, but remain "
    "decision-support signals — all actions pass through the compliance guardrail layer and "
    "must operate under approved, regulator-reviewed collection policies."
)
