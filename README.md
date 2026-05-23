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
FDA Adverse Event Reporting System (FAERS)

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
