
import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 2.4rem;
    margin-bottom: 8px;
}

.hero p {
    font-size: 1.05rem;
    color: #d1d5db;
}

.card {
    padding: 1.5rem;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.metric-card {
    padding: 1.4rem;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.metric-title {
    font-size: 0.9rem;
    color: #6b7280;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 5px;
}

.risk-low {
    padding: 1.5rem;
    border-radius: 15px;
    background-color: #ecfdf5;
    border: 1px solid #10b981;
    color: #065f46;
}

.risk-high {
    padding: 1.5rem;
    border-radius: 15px;
    background-color: #fef2f2;
    border: 1px solid #ef4444;
    color: #991b1b;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("models/xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("models/threshold.pkl", "rb") as file:
        threshold = pickle.load(file)

    return model, threshold


model, threshold = load_model()


st.markdown("""
<div class="hero">

<h1>💳 Credit Card Default Risk Predictor</h1>

<p>
Machine Learning powered credit-risk assessment using a tuned XGBoost classification model.
</p>

</div>
""", unsafe_allow_html=True)


st.sidebar.title("Customer Information")
st.sidebar.markdown("Enter the customer's financial and demographic information.")


st.sidebar.subheader("Personal Information")

LIMIT_BAL = st.sidebar.number_input(
    "Credit Limit",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

SEX = st.sidebar.selectbox(
    "Sex",
    [1, 2]
)

EDUCATION = st.sidebar.selectbox(
    "Education",
    [1, 2, 3, 4, 5, 6]
)

MARRIAGE = st.sidebar.selectbox(
    "Marriage",
    [1, 2, 3]
)

AGE = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)


st.sidebar.subheader("Payment Status")

PAY_1 = st.sidebar.number_input("PAY_1", value=0, step=1)
PAY_2 = st.sidebar.number_input("PAY_2", value=0, step=1)
PAY_3 = st.sidebar.number_input("PAY_3", value=0, step=1)
PAY_4 = st.sidebar.number_input("PAY_4", value=0, step=1)
PAY_5 = st.sidebar.number_input("PAY_5", value=0, step=1)
PAY_6 = st.sidebar.number_input("PAY_6", value=0, step=1)


st.sidebar.subheader("Bill Amounts")

BILL_AMT1 = st.sidebar.number_input("BILL_AMT1", value=5000.0)
BILL_AMT2 = st.sidebar.number_input("BILL_AMT2", value=5000.0)
BILL_AMT3 = st.sidebar.number_input("BILL_AMT3", value=5000.0)
BILL_AMT4 = st.sidebar.number_input("BILL_AMT4", value=5000.0)
BILL_AMT5 = st.sidebar.number_input("BILL_AMT5", value=5000.0)
BILL_AMT6 = st.sidebar.number_input("BILL_AMT6", value=5000.0)


st.sidebar.subheader("Payment Amounts")

PAY_AMT1 = st.sidebar.number_input("PAY_AMT1", value=2000.0)
PAY_AMT2 = st.sidebar.number_input("PAY_AMT2", value=2000.0)
PAY_AMT3 = st.sidebar.number_input("PAY_AMT3", value=2000.0)
PAY_AMT4 = st.sidebar.number_input("PAY_AMT4", value=2000.0)
PAY_AMT5 = st.sidebar.number_input("PAY_AMT5", value=2000.0)
PAY_AMT6 = st.sidebar.number_input("PAY_AMT6", value=2000.0)


PAY_STATUS = [
    PAY_1,
    PAY_2,
    PAY_3,
    PAY_4,
    PAY_5,
    PAY_6
]

BILL_AMOUNTS = [
    BILL_AMT1,
    BILL_AMT2,
    BILL_AMT3,
    BILL_AMT4,
    BILL_AMT5,
    BILL_AMT6
]

PAY_AMOUNTS = [
    PAY_AMT1,
    PAY_AMT2,
    PAY_AMT3,
    PAY_AMT4,
    PAY_AMT5,
    PAY_AMT6
]


NUM_DELAYED_MONTHS = sum(x > 0 for x in PAY_STATUS)

MAX_DELAY = max(PAY_STATUS)

TOTAL_PAY_AMT = sum(PAY_AMOUNTS)

AVG_PAY_AMT = np.mean(PAY_AMOUNTS)

MAX_PAY_AMT = max(PAY_AMOUNTS)

ZERO_PAY_MONTHS = sum(x == 0 for x in PAY_AMOUNTS)

TOTAL_BILL_AMT = sum(BILL_AMOUNTS)

AVG_BILL_AMT = np.mean(BILL_AMOUNTS)

MAX_BILL_AMT = max(BILL_AMOUNTS)

BILL_CHANGE_RECENT = BILL_AMT1 - BILL_AMT2

PAY_CHANGE_RECENT = PAY_1 - PAY_2

PAY_BILL_RATIO = TOTAL_PAY_AMT / (abs(TOTAL_BILL_AMT) + 1)


new_customer = pd.DataFrame([{

    "LIMIT_BAL": LIMIT_BAL,
    "SEX": SEX,
    "EDUCATION": EDUCATION,
    "MARRIAGE": MARRIAGE,
    "AGE": AGE,

    "PAY_1": PAY_1,
    "PAY_2": PAY_2,
    "PAY_3": PAY_3,
    "PAY_4": PAY_4,
    "PAY_5": PAY_5,
    "PAY_6": PAY_6,

    "BILL_AMT1": BILL_AMT1,
    "BILL_AMT2": BILL_AMT2,
    "BILL_AMT3": BILL_AMT3,
    "BILL_AMT4": BILL_AMT4,
    "BILL_AMT5": BILL_AMT5,
    "BILL_AMT6": BILL_AMT6,

    "PAY_AMT1": PAY_AMT1,
    "PAY_AMT2": PAY_AMT2,
    "PAY_AMT3": PAY_AMT3,
    "PAY_AMT4": PAY_AMT4,
    "PAY_AMT5": PAY_AMT5,
    "PAY_AMT6": PAY_AMT6,

    "NUM_DELAYED_MONTHS": NUM_DELAYED_MONTHS,
    "MAX_DELAY": MAX_DELAY,
    "TOTAL_PAY_AMT": TOTAL_PAY_AMT,
    "AVG_PAY_AMT": AVG_PAY_AMT,
    "MAX_PAY_AMT": MAX_PAY_AMT,
    "ZERO_PAY_MONTHS": ZERO_PAY_MONTHS,
    "TOTAL_BILL_AMT": TOTAL_BILL_AMT,
    "AVG_BILL_AMT": AVG_BILL_AMT,
    "MAX_BILL_AMT": MAX_BILL_AMT,
    "BILL_CHANGE_RECENT": BILL_CHANGE_RECENT,
    "PAY_CHANGE_RECENT": PAY_CHANGE_RECENT,
    "PAY_BILL_RATIO": PAY_BILL_RATIO

}])


st.markdown(
    '<div class="section-title">Prediction Dashboard</div>',
    unsafe_allow_html=True
)


if st.button(
    "🔍 Assess Default Risk",
    type="primary",
    use_container_width=True
):

    probability = model.predict_proba(new_customer)[0, 1]

    prediction = int(probability >= threshold)

    probability_percent = probability * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Default Probability</div>
            <div class="metric-value">{probability_percent:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Decision Threshold</div>
            <div class="metric-value">{threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        if prediction == 1:
            decision = "Likely Default"
        else:
            decision = "Unlikely to Default"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Prediction</div>
            <div class="metric-value">{decision}</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("### Risk Assessment")

    if prediction == 1:

        st.markdown(f"""
        <div class="risk-high">

        <h3>⚠️ Higher Default Risk</h3>

        <p>
        The model estimates a default probability of
        <strong>{probability_percent:.2f}%</strong>.
        </p>

        <p>
        This probability is above the optimized decision threshold
        of <strong>{threshold:.2f}</strong>.
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="risk-low">

        <h3>✅ Lower Default Risk</h3>

        <p>
        The model estimates a default probability of
        <strong>{probability_percent:.2f}%</strong>.
        </p>

        <p>
        This probability is below the optimized decision threshold
        of <strong>{threshold:.2f}</strong>.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.progress(
        min(float(probability), 1.0),
        text=f"Estimated Default Probability: {probability_percent:.2f}%"
    )


    with st.expander("View Customer Features"):

        st.dataframe(
            new_customer.T.rename(columns={0: "Value"}),
            use_container_width=True
        )


st.markdown("---")


col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="card">

    <h3>🤖 Model Information</h3>

    <p><strong>Algorithm:</strong> Tuned XGBoost Classifier</p>

    <p><strong>Problem:</strong> Binary Classification</p>

    <p><strong>Output:</strong> Default Probability</p>

    <p><strong>Decision Threshold:</strong> Optimized using F1-score</p>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    <h3>📊 Project Highlights</h3>

    <p>• Exploratory Data Analysis</p>

    <p>• Feature Engineering</p>

    <p>• Multiple Model Comparison</p>

    <p>• Stratified Cross-Validation</p>

    <p>• Hyperparameter Tuning</p>

    <p>• Threshold Optimization</p>

    </div>
    """, unsafe_allow_html=True)


st.markdown("---")

st.caption(
    "Credit Card Default Prediction | Machine Learning Project | XGBoost + Streamlit"
)
