# Drug-Interaction-Optimization-using-Quantum-Simulation
Quantum-powered drug interaction risk analysis using AI and quantum simulation. (QuantumX Hackathon). A hybrid AI + Quantum Computing prototype that predicts and analyzes potential drug-drug interaction risks using real-world healthcare datasets and quantum feature simulation.

---

# 📌 Problem Statement

Drug-drug interactions can lead to:
- severe adverse effects
- emergency hospitalizations
- treatment complications
- delayed medical decisions

Traditional systems struggle to efficiently analyze complex combinations of medications and side effects at scale.
This project explores how **Quantum Computing techniques** can enhance healthcare analytics and interaction prediction.

---

# 🌍 SDG Alignment

## SDG 3 — Good Health & Well-being

This project contributes toward:
- safer medication usage
- healthcare decision support
- intelligent adverse effect analysis
- improved patient safety

---

# 🚀 Project Objective

Build a prototype that:
- predicts interaction risk between drugs
- analyzes adverse effects
- compares classical ML with quantum-enhanced predictions
- demonstrates practical healthcare applications of quantum simulation

---

# 🧠 Solution Overview

The system combines:

## 🤖 Classical Machine Learning
- Random Forest Classifier
- Baseline risk prediction

## ⚛️ Quantum Feature Simulation
Using:
- PennyLane
- Qubits
- Quantum circuits
- Entanglement layers

The project uses a hybrid workflow where:
1. Classical ML learns interaction patterns
2. Quantum circuits generate enhanced feature representations
3. Predictions are compared for analysis

---

# 📊 Dataset Used

## TWOSIDES Dataset
Source: 
[nSIDES - Drug side effect and interaction resources](https://nsides.io/#offsides-and-twosides) 

Dataset contains:
- Drug 1
- Drug 2
- Side Effects
- PRR Scores
- Adverse Event Frequencies

---

# 🏗️ Project Architecture

```text
TWOSIDES Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Encoding
        ↓
Classical ML Model
        ↓
Quantum Circuit Simulation
        ↓
Risk Prediction
        ↓
Interactive Streamlit Dashboard
```

# 📁 Project Structure

```text
QuantumX Hackathon/
│
├── app/
│   └── app.py                     # Streamlit web application
│
├── data/ (Downloaded from the website not uploaded in github)
│   └── raw/
│       ├── OFFSIDES.csv           # Single-drug side effects dataset
│       └── TWOSIDES.csv           # Drug-drug interaction dataset
│
├── models/
│   ├── rf_model.pkl               # Classical ML model
│   ├── quantum_model.pkl          # Quantum-enhanced model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder.pkl          # Risk label encoder
│   ├── drug1_encoder.pkl          # Drug 1 encoder
│   ├── drug2_encoder.pkl          # Drug 2 encoder
│   ├── effect_encoder.pkl         # Side effect encoder
│   └── demo_data.pkl              # Sample processed dataset
│
├── notebooks/
│   └── 01_explore_data.ipynb                  # Exploratory Data Analysis
|   └── 02_baseline_ml.ipynb
|   └── 03_quantum_model.ipynb 
│
├── src/
│   └── train_models.py            # Training pipeline
│
├── requirements.txt               # Python dependencies
│
└── README.md
```

# 🔮 Future Scope

- Support analysis of 3+ drug combinations and complex prescription chains.
- Recommend safer alternative medicines based on patient allergies.
- Enable personalized medicine using patient history and clinical data.
- Integrate with real quantum hardware like IBM Quantum and Amazon Braket.
- Improve predictions using Quantum Neural Networks (QNNs).
- Connect with hospital systems for real-time prescription validation.
- Expand into scalable AI-driven healthcare decision support systems.
