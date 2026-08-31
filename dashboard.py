import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
# ============================================================
# CONFIG
# ============================================================

#API_URL = os.getenv("API_URL","http://127.0.0.1:8000",).rstrip("/")
API_URL = os.getenv("API_URL","https://debt-collection-api-455207252491.asia-south1.run.app",).rstrip("/")


st.set_page_config(
    page_title="Debt Collection Intelligence",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# SAMPLE DATA
# ============================================================

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customer_id": [f"C{1000+i}" for i in range(n)],
        "customer_name": [
        f"Customer {1000+i}" for i in range(n)],
        "age": np.random.randint(21, 70, n),
        "income": np.random.randint(
            20000,
            150000,
            n
        ),

        "loan_amount": np.random.randint(
            50000,
            1500000,
            n
        ),

        "outstanding_amount": np.random.randint(
            10000,
            800000,
            n
        ),

        "emi_amount": np.random.randint(
            2000,
            50000,
            n
        ),

        "credit_score": np.random.randint(
            450,
            850,
            n
        ),

        "dpd": np.random.choice(
            [0, 5, 15, 30, 45, 60, 90, 120, 180],
            n
        ),

        "missed_payment_count": np.random.randint(
            0,
            7,
            n
        ),

        "previous_ptp_count": np.random.randint(
            0,
            7,
            n
        ),

        "previous_ptp_kept_count": np.random.randint(
            0,
            6,
            n
        ),

        "collection_attempts": np.random.randint(
            0,
            12,
            n
        ),

        "successful_contacts": np.random.randint(
            0,
            8,
            n
        ),

        "recent_payment_amount": np.random.randint(
            0,
            30000,
            n
        ),

        "days_since_last_payment": np.random.randint(
            0,
            180,
            n
        ),

        "preferred_channel": np.random.choice(
            [
                "PHONE",
                "SMS",
                "WHATSAPP",
                "EMAIL"
            ],
            n
        )
    })
    df["due_date"] = (
                        pd.Timestamp.today().normalize()
                        - pd.to_timedelta(df["dpd"], unit="D")
                    ).dt.strftime("%d-%m-%Y")
    # --------------------------------------------------------
    # Simulated AI predictions
    # --------------------------------------------------------

    stress = (
        0.40 * np.minimum(df["dpd"] / 180, 1)
        +
        0.20 * np.minimum(
            df["missed_payment_count"] / 6,
            1
        )
        +
        0.20 * np.minimum(
            df["days_since_last_payment"] / 180,
            1
        )
        +
        0.20 * (
            1 -
            (
                df["previous_ptp_kept_count"]
                /
                np.maximum(
                    df["previous_ptp_count"],
                    1
                )
            )
        )
    )

    df["payment_probability"] = np.clip(
        0.95 - stress + np.random.normal(
            0,
            0.05,
            n
        ),
        0.02,
        0.98
    )

    df["ptp_probability"] = np.clip(
        0.90 - stress * 0.9 + np.random.normal(
            0,
            0.05,
            n
        ),
        0.02,
        0.98
    )

    df["recovery_probability"] = np.clip(
        0.85 - stress * 0.7 + np.random.normal(
            0,
            0.05,
            n
        ),
        0.02,
        0.98
    )

    # --------------------------------------------------------
    # Collection priority
    # --------------------------------------------------------

    df["priority_score"] = (
        (1 - df["payment_probability"]) * 40
        +
        (1 - df["ptp_probability"]) * 25
        +
        np.minimum(df["dpd"] / 180, 1) * 25
        +
        np.minimum(
            df["outstanding_amount"] /
            np.maximum(df["income"] * 12, 1),
            1
        ) * 10
    )

    df["priority_score"] = np.clip(
        df["priority_score"],
        0,
        100
    )

    def priority(x):

        if x >= 70:
            return "HIGH"

        elif x >= 40:
            return "MEDIUM"

        return "LOW"

    df["priority"] = df["priority_score"].apply(
        priority
    )

    # --------------------------------------------------------
    # Next Best Action
    # --------------------------------------------------------

    def nba(row):

        if row["dpd"] >= 90:
            return "Human Review"

        if row["ptp_probability"] >= 0.65:
            return "Request PTP"

        if row["payment_probability"] >= 0.60:
            return "Repayment Discussion"
        return "Collection Call"
    df["next_best_action"] = df.apply(
        nba,
        axis=1
    )

    return df


df = load_data()

# ============================================================
# HEADER
# ============================================================

st.title(
    "💳 AI Debt Collection Intelligence Dashboard"
)

st.caption(
    "ML + Next Best Action + RAG + GenAI"
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔎 Filters"
)

priority_filter = st.sidebar.multiselect(
    "Priority",
    options=[
        "HIGH",
        "MEDIUM",
        "LOW"
    ],
    default=[
        "HIGH",
        "MEDIUM",
        "LOW"
    ]
)

dpd_filter = st.sidebar.slider(
    "Minimum DPD",
    min_value=0,
    max_value=180,
    value=0
)

channel_filter = st.sidebar.multiselect(
    "Preferred Channel",
    options=df[
        "preferred_channel"
    ].unique(),
    default=list(
        df[
            "preferred_channel"
        ].unique()
    )
)

filtered_df = df[
    (df["priority"].isin(priority_filter))
    &
    (df["dpd"] >= dpd_filter)
    &
    (df["preferred_channel"].isin(channel_filter))
]

# ============================================================
# KPI SECTION
# ============================================================

total_customers = len(filtered_df)

total_outstanding = (
    filtered_df[
        "outstanding_amount"
    ].sum()
)

high_risk = len(
    filtered_df[
        filtered_df["priority"] == "HIGH"
    ]
)

avg_payment_probability = (
    filtered_df[
        "payment_probability"
    ].mean()
)

avg_recovery_probability = (
    filtered_df[
        "recovery_probability"
    ].mean()
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Outstanding",
    f"₹{total_outstanding/1e7:.2f} Cr"
)

col3.metric(
    "High Risk",
    f"{high_risk:,}"
)

col4.metric(
    "Avg Payment Probability",
    f"{avg_payment_probability*100:.1f}%"
)

col5.metric(
    "Avg Recovery Probability",
    f"{avg_recovery_probability*100:.1f}%"
)

st.divider()

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "📊 AI Risk Distribution"
)

col1, col2 = st.columns(2)

with col1:

    risk_counts = (
        filtered_df[
            "priority"
        ]
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )

with col2:

    action_counts = (
        filtered_df[
            "next_best_action"
        ]
        .value_counts()
    )

    st.bar_chart(
        action_counts
    )

# ============================================================
# AI PREDICTIONS
# ============================================================

st.subheader(
    "🤖 AI Prediction Summary"
)

prediction_data = pd.DataFrame({
    "Metric": [
        "Payment Probability",
        "PTP Probability",
        "Recovery Probability"
    ],

    "Probability": [
        avg_payment_probability,
        filtered_df[
            "ptp_probability"
        ].mean(),

        avg_recovery_probability
    ]
})

prediction_data[
    "Probability"
] = prediction_data[
    "Probability"
] * 100

st.dataframe(
    prediction_data.style.format({
        "Probability": "{:.1f}%"
    }),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# CUSTOMER PRIORITY TABLE
# ============================================================

st.subheader(
    "🚨 AI Collection Priority Queue"
)

display_columns = [
    "customer_id",
    "dpd",
    "outstanding_amount",
    "payment_probability",
    "ptp_probability",
    "recovery_probability",
    "priority_score",
    "priority",
    "next_best_action",
    "preferred_channel"
]

display_df = (
    filtered_df[
        display_columns
    ]
    .sort_values(
        "priority_score",
        ascending=False
    )
    .head(50)
    .copy()
)

display_df[
    "payment_probability"
] *= 100

display_df[
    "ptp_probability"
] *= 100

display_df[
    "recovery_probability"
] *= 100

st.dataframe(
    display_df.style.format({
        "outstanding_amount": "₹{:,.0f}",
        "payment_probability": "{:.1f}%",
        "ptp_probability": "{:.1f}%",
        "recovery_probability": "{:.1f}%",
        "priority_score": "{:.1f}"
    }),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# CUSTOMER 360
# ============================================================

st.divider()

st.subheader(
    "👤 Customer 360"
)

customer_id = st.selectbox(
    "Select Customer",
    filtered_df[
        "customer_id"
    ].tolist()
)

customer = filtered_df[
    filtered_df[
        "customer_id"
    ] == customer_id
].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:

    st.write(
        "### Customer Information"
    )

    st.write(
        f"**Customer ID:** {customer['customer_id']}"
    )
    st.write(
        f"**Customer Name:** {customer['customer_name']}"
    )
    st.write(
        f"**Age:** {customer['age']}"
    )

    st.write(
        f"**Income:** ₹{customer['income']:,.0f}"
    )

    st.write(
        f"**Credit Score:** {customer['credit_score']}"
    )

with col2:

    st.write(
        "### Loan Information"
    )

    st.write(
        f"**Loan Amount:** ₹{customer['loan_amount']:,.0f}"
    )
    st.write(
            f"**Due Date:** {customer['due_date']}"
        )
    st.write(
        f"**Outstanding:** ₹{customer['outstanding_amount']:,.0f}"
    )

    st.write(
        f"**EMI:** ₹{customer['emi_amount']:,.0f}"
    )

    st.write(
        f"**DPD:** {customer['dpd']}"
    )

with col3:

    st.write(
        "### AI Insights"
    )

    st.metric(
        "Payment Probability",
        f"{customer['payment_probability']*100:.1f}%"
    )

    st.metric(
        "PTP Probability",
        f"{customer['ptp_probability']*100:.1f}%"
    )

    st.metric(
        "Recovery Probability",
        f"{customer['recovery_probability']*100:.1f}%"
    )

# ============================================================
# NEXT BEST ACTION
# ============================================================

st.subheader(
    "🎯 AI Next Best Action"
)

nba = customer[
    "next_best_action"
]

channel = customer[
    "preferred_channel"
]

priority = customer[
    "priority"
]

if priority == "HIGH":

    st.error(
        f"🚨 HIGH PRIORITY CUSTOMER"
    )

elif priority == "MEDIUM":

    st.warning(
        f"⚠️ MEDIUM PRIORITY CUSTOMER"
    )

else:

    st.success(
        f"✅ LOW PRIORITY CUSTOMER"
    )

col1, col2, col3 = st.columns(3)

col1.metric(
    "Priority Score",
    f"{customer['priority_score']:.1f}"
)

col2.metric(
    "Recommended Channel",
    channel
)

col3.metric(
    "Next Best Action",
    nba
)

st.info(
    f"""
    **AI Recommendation**

    Customer {customer_id} should be handled using:

    **Action:** {nba}

    **Channel:** {channel}

    **Priority:** {priority}

    **Reason:** DPD = {customer['dpd']} days,
    Payment Probability = {customer['payment_probability']*100:.1f}%,
    PTP Probability = {customer['ptp_probability']*100:.1f}%.
    """
)

# ============================================================
# RAG ASSISTANT
# ============================================================

st.divider()

st.subheader(
    "📚 AI Collection Policy Assistant"
)

question = st.text_input(
    "Ask a question about collection policy"
)

if st.button(
    "Ask AI Policy Assistant"
):

    if question.strip():

        try:

            response = requests.post(
                f"{API_URL}/rag/query",
                json={
                    "question": question
                },
                timeout=60
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    result.get(
                        "answer",
                        "No answer"
                    )
                )

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    st.write(
                        "### Sources"
                    )

                    for source in sources:

                        st.write(
                            f"📄 {source.get('title')}"
                        )

            else:

                st.error(
                    response.text
                )

        except Exception as e:

            st.error(
                f"Could not connect to FastAPI: {e}"
            )

# ============================================================
# GENAI MESSAGE
# ============================================================

st.divider()

st.subheader(
    "✍️ GenAI Collection Message"
)

tone = st.selectbox(
    "Message Tone",
    [
        "professional",
        "empathetic",
        "concise"
    ]
)

current_customer_id = str(customer["customer_id"])

# Never reuse a generated message for a different customer.
if st.session_state.get("message_customer_id") != current_customer_id:
    st.session_state["message_customer_id"] = current_customer_id
    st.session_state["generated_message"] = ""

if st.button(
    "Generate Collection Message"
):

    payload = {

        "customer": {

            "customer_id":
                customer["customer_id"],
            "customer_name": str(customer["customer_name"]),
            "due_date": str(customer["due_date"]),

            "age":
                int(customer["age"]),

            "income":
                float(customer["income"]),

            "loan_amount":
                float(customer["loan_amount"]),

            "outstanding_amount":
                float(customer["outstanding_amount"]),

            "emi_amount":
                float(customer["emi_amount"]),

            "credit_score":
                float(customer["credit_score"]),

            "dpd":
                int(customer["dpd"]),

            "missed_payment_count":
                int(
                    customer[
                        "missed_payment_count"
                    ]
                ),

            "previous_ptp_count":
                int(
                    customer[
                        "previous_ptp_count"
                    ]
                ),

            "previous_ptp_kept_count":
                int(
                    customer[
                        "previous_ptp_kept_count"
                    ]
                ),

            "collection_attempts":
                int(
                    customer[
                        "collection_attempts"
                    ]
                ),

            "successful_contacts":
                int(
                    customer[
                        "successful_contacts"
                    ]
                ),

            "recent_payment_amount":
                float(
                    customer[
                        "recent_payment_amount"
                    ]
                ),

            "days_since_last_payment":
                int(
                    customer[
                        "days_since_last_payment"
                    ]
                ),

            "preferred_channel":
                customer[
                    "preferred_channel"
                ]
        },

        "tone": tone
    }

    try:
        response = requests.post(
            f"{API_URL}/generate-message",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:

            result = response.json()

            generated_message = result.get("message", "").strip()

            if generated_message:
                st.session_state["generated_message"] = generated_message
            else:
                st.error("FastAPI returned an empty message.")

        else:

            st.error(
                response.text
            )

    except Exception as e:

        st.error(
            f"Could not connect to FastAPI: {e}"
        )

# ============================================================
# DELIVERY CHANNEL PLACEHOLDERS
# ============================================================

if st.session_state.get("generated_message"):

    st.write("### AI Generated Message")
    st.info(st.session_state["generated_message"])

    mail_column, whatsapp_column = st.columns(2)

    with mail_column:
        if st.button("📧 Mail", use_container_width=True):
            st.info("Mail option selected. Sending will be added later.")

    with whatsapp_column:
        if st.button("💬 WhatsApp", use_container_width=True):
            st.info("WhatsApp option selected. Sending will be added later.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Debt Collection Intelligence Platform | "
    "ML predictions are decision-support signals and "
    "must operate under approved collection policies."
)
