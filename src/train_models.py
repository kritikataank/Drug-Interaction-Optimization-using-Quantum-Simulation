import pandas as pd
import numpy as np
import pennylane as qml
import joblib

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------
TARGET_ROWS = 500

# ------------------------------------------------
# LOAD SMALL SAMPLE FROM HUGE CSV
# ------------------------------------------------
print("Loading sampled dataset...")

cols = [
    "drug_1_concept_name",
    "drug_2_concept_name",
    "condition_concept_name",
    "PRR"
]

chunks = []

chunk_iter = pd.read_csv(
    "../data/raw/TWOSIDES.csv",
    usecols=cols,
    chunksize=100000,
    low_memory=False
)

for chunk in chunk_iter:

    chunk["PRR"] = pd.to_numeric(
        chunk["PRR"],
        errors="coerce"
    )

    chunk = chunk.dropna()

    # Take tiny sample from each chunk
    sample_chunk = chunk.sample(
        n=min(50, len(chunk)),
        random_state=42
    )

    chunks.append(sample_chunk)

    current_rows = sum(len(c) for c in chunks)

    print(f"Collected rows: {current_rows}")

    if current_rows >= TARGET_ROWS:
        break

# Combine all samples
df = pd.concat(chunks)

# Final sample size control
df = df.sample(
    n=min(TARGET_ROWS, len(df)),
    random_state=42
)

print("Final dataset shape:", df.shape)

# ------------------------------------------------
# CREATE LABELS
# ------------------------------------------------
def risk_label(prr):

    if prr < 1.5:
        return "Low"

    elif prr < 3:
        return "Medium"

    else:
        return "High"


df["risk_level"] = df["PRR"].apply(risk_label)

# ------------------------------------------------
# ENCODERS
# ------------------------------------------------
drug1_encoder = LabelEncoder()
drug2_encoder = LabelEncoder()
effect_encoder = LabelEncoder()
label_encoder = LabelEncoder()

df["drug1_encoded"] = drug1_encoder.fit_transform(
    df["drug_1_concept_name"]
)

df["drug2_encoded"] = drug2_encoder.fit_transform(
    df["drug_2_concept_name"]
)

df["effect_encoded"] = effect_encoder.fit_transform(
    df["condition_concept_name"]
)

df["label"] = label_encoder.fit_transform(
    df["risk_level"]
)

# ------------------------------------------------
# FEATURES
# ------------------------------------------------
X = df[
    ["drug1_encoded", "drug2_encoded", "effect_encoded"]
]

y = df["label"]

# ------------------------------------------------
# CLASSICAL MODEL
# ------------------------------------------------
print("Training RandomForest...")

rf_model = RandomForestClassifier(
    n_estimators=20,
    random_state=42
)

rf_model.fit(X, y)

print("RandomForest ready")

# ------------------------------------------------
# SCALE FEATURES
# ------------------------------------------------
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# ------------------------------------------------
# QUANTUM DEVICE
# ------------------------------------------------
n_qubits = 3

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)

# ------------------------------------------------
# QUANTUM CIRCUIT
# ------------------------------------------------
@qml.qnode(dev)
def quantum_circuit(inputs):

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]

# ------------------------------------------------
# QUANTUM FEATURES
# ------------------------------------------------
print("Generating quantum features...")

quantum_features = np.array([
    quantum_circuit(x)
    for x in X_scaled
])

print("Quantum features ready")

# ------------------------------------------------
# QUANTUM MODEL
# ------------------------------------------------
print("Training quantum model...")

quantum_model = LogisticRegression(
    max_iter=1000
)

quantum_model.fit(
    quantum_features,
    y
)

print("Quantum model ready")

# ------------------------------------------------
# SAVE MODELS
# ------------------------------------------------
print("Saving models...")

joblib.dump(
    rf_model,
    "../models/rf_model.pkl"
)

joblib.dump(
    quantum_model,
    "../models/quantum_model.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

joblib.dump(
    drug1_encoder,
    "../models/drug1_encoder.pkl"
)

joblib.dump(
    drug2_encoder,
    "../models/drug2_encoder.pkl"
)

joblib.dump(
    effect_encoder,
    "../models/effect_encoder.pkl"
)

joblib.dump(
    label_encoder,
    "../models/label_encoder.pkl"
)

joblib.dump(
    df,
    "../models/demo_data.pkl"
)

print("✅ ALL MODELS SAVED SUCCESSFULLY")