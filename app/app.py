import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pennylane as qml

# =====================================================
# PATHS
# =====================================================
MODELS_PATH = r"D:\QuantumX Hackathon\models"

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Quantum Drug Interaction Analyzer",
    layout="wide"
)

st.title("⚛️ Quantum Drug Interaction Analyzer")

st.markdown("""
This prototype predicts drug interaction risk levels using:

- 🤖 Classical Machine Learning
- ⚛️ Quantum Feature Simulation
""")

# =====================================================
# LOAD SAVED MODELS
# =====================================================
@st.cache_resource
def load_models():

    rf_model = joblib.load(
        rf"{MODELS_PATH}\rf_model.pkl"
    )

    quantum_model = joblib.load(
        rf"{MODELS_PATH}\quantum_model.pkl"
    )

    scaler = joblib.load(
        rf"{MODELS_PATH}\scaler.pkl"
    )

    drug1_encoder = joblib.load(
        rf"{MODELS_PATH}\drug1_encoder.pkl"
    )

    drug2_encoder = joblib.load(
        rf"{MODELS_PATH}\drug2_encoder.pkl"
    )

    effect_encoder = joblib.load(
        rf"{MODELS_PATH}\effect_encoder.pkl"
    )

    label_encoder = joblib.load(
        rf"{MODELS_PATH}\label_encoder.pkl"
    )

    demo_data = joblib.load(
        rf"{MODELS_PATH}\demo_data.pkl"
    )

    return (
        rf_model,
        quantum_model,
        scaler,
        drug1_encoder,
        drug2_encoder,
        effect_encoder,
        label_encoder,
        demo_data
    )

# =====================================================
# LOAD EVERYTHING
# =====================================================
(
    rf_model,
    quantum_model,
    scaler,
    drug1_encoder,
    drug2_encoder,
    effect_encoder,
    label_encoder,
    demo_data
) = load_models()

# =====================================================
# QUANTUM DEVICE
# =====================================================
n_qubits = 3

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)

# =====================================================
# QUANTUM CIRCUIT
# =====================================================
@qml.qnode(dev)
def quantum_circuit(inputs):

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🧪 Interaction Inputs")

drug1 = st.sidebar.selectbox(
    "Select Drug 1",
    sorted(
        demo_data["drug_1_concept_name"].unique()
    )
)

drug2 = st.sidebar.selectbox(
    "Select Drug 2",
    sorted(
        demo_data["drug_2_concept_name"].unique()
    )
)

effect = st.sidebar.selectbox(
    "Possible Side Effect",
    sorted(
        demo_data["condition_concept_name"].unique()
    )
)

# =====================================================
# PREDICTION
# =====================================================
if st.sidebar.button("Predict Risk"):

    # -----------------------------------------
    # Encode Inputs
    # -----------------------------------------
    d1 = drug1_encoder.transform([drug1])[0]
    d2 = drug2_encoder.transform([drug2])[0]
    eff = effect_encoder.transform([effect])[0]

    input_df = pd.DataFrame(
        [[d1, d2, eff]],
        columns=[
            "drug1_encoded",
            "drug2_encoded",
            "effect_encoded"
        ]
    )

    # -----------------------------------------
    # Classical Prediction
    # -----------------------------------------
    classical_pred = rf_model.predict(
        input_df
    )[0]

    classical_label = label_encoder.inverse_transform(
        [classical_pred]
    )[0]

    # -----------------------------------------
    # Quantum Prediction
    # -----------------------------------------
    scaled_input = scaler.transform(
        input_df
    )

    quantum_features = np.array([
        quantum_circuit(scaled_input[0])
    ])

    quantum_pred = quantum_model.predict(
        quantum_features
    )[0]

    quantum_label = label_encoder.inverse_transform(
        [quantum_pred]
    )[0]

    # =================================================
    # RESULTS
    # =================================================
    st.subheader("🧪 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Classical ML Prediction",
            classical_label
        )

    with col2:
        st.metric(
            "Quantum Prediction",
            quantum_label
        )

    # -----------------------------------------
    # Risk Status
    # -----------------------------------------
    if quantum_label == "High":

        st.error(
            "⚠️ High Risk Drug Interaction"
        )

    elif quantum_label == "Medium":

        st.warning(
            "⚠️ Medium Risk Interaction"
        )

    else:

        st.success(
            "✅ Low Risk Interaction"
        )

    # -----------------------------------------
    # Selected Details
    # -----------------------------------------
    st.markdown("---")

    st.subheader("📋 Selected Combination")

    st.write(f"**Drug 1:** {drug1}")
    st.write(f"**Drug 2:** {drug2}")
    st.write(f"**Potential Side Effect:** {effect}")

# =====================================================
# DATA PREVIEW
# =====================================================
st.markdown("---")

st.subheader("📊 Training Sample Preview")

st.dataframe(
    demo_data.head(10)
)

st.write(
    f"Dataset Shape: {demo_data.shape}"
)