
# 🛡️ SecurePay – Real-Time Fraud Detection Engine

Production-grade fraud detection system leveraging **unsupervised anomaly detection** to identify high-risk financial transactions in real time.

🔗 **Live API Docs:** https://securepay-fraud-detection.onrender.com/docs

---

## 📌 Overview

SecurePay is designed to detect fraudulent transactions in highly imbalanced financial datasets (~0.8% fraud rate) using **behavioral anomaly detection** instead of traditional supervised classification.

The system models *normal transaction behavior* and flags deviations as potential fraud, making it robust to **unknown and evolving attack patterns**.

---

## 🚀 Key Features

- ⚡ **Real-Time Fraud Detection API**
- 🧠 **Unsupervised Learning (Isolation Forest)**
- 📊 **Risk-Based Transaction Segmentation**
- 📉 Handles **Extreme Class Imbalance**
- 🔍 Detects **Previously Unseen Fraud Patterns**
- 📦 Fully deployable ML pipeline

---

## 🏗️ System Architecture


Client Request → FastAPI Backend → Preprocessing → Model Inference → Risk Scoring → Response


---

## 🤖 Model Details

### Primary Model: Isolation Forest

- `contamination = 0.02`
- `n_estimators = 300`
- `max_samples = 512`

### Why Isolation Forest?

- Scales efficiently to large datasets  
- Suitable for real-time inference  
- Robust to distribution shifts  
- Detects anomalies via feature isolation  

---

## 📊 Performance Summary

| Metric | Value |
|------|------|
| Precision | ~0.35 – 0.41 |
| Recall | ~0.56 – 0.61 |
| F1 Score | ~0.44 |

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|--|----------------|----------------|
| **Actual Normal** | 56,523 | 340 |
| **Actual Fraud** | 258 | 234 |

---

## 🎯 Risk-Based Decision System

Instead of binary classification, SecurePay uses **risk segmentation**:

| Score Range | Risk Level | Action |
|------------|-----------|--------|
| Low | ✅ Allow | No action |
| Medium | ⚠️ Alert | Soft warning |
| High | 🔍 Review | Manual verification |
| Critical | ⛔ Block | Immediate decline |

### Observed Fraud Rates

- LOW → ~0.14%  
- MEDIUM → ~6.5%  
- HIGH → ~28%  
- CRITICAL → ~64%  

---

## 📡 API Usage

### 🔗 Base URL


https://securepay-fraud-detection.onrender.com


---

### 📌 Endpoint: Predict Fraud Risk


POST /predict


### 📥 Request Example

```json
{
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  ...
  "Amount_log": 3.2,
  "Hour": 14
}
📤 Response Example
{
  "anomaly_score": 0.78,
  "risk_level": "CRITICAL",
  "action": "BLOCK"
}
🧪 How It Works
Input transaction features are received via API
Data is preprocessed and transformed
Isolation Forest computes anomaly score
Score is normalized to [0,1]
Transaction is assigned a risk bucket
Action is returned (Allow / Alert / Review / Block)

📁 Project Structure
securepay-fraud-detection/
│
├── data/
├── notebooks/
├── src/
├── models/
├── frontend/
├── outputs/
└── README.md

⚙️ Local Setup
git clone https://github.com/your-username/securepay-fraud-detection.git
cd securepay-fraud-detection

pip install -r requirements.txt

uvicorn src.main:app --reload

🌐 Deployment
Backend: FastAPI
Hosting: Render
API Docs: /docs (Swagger UI)

📈 Business Impact
Reduces fraud losses through early detection
Minimizes customer friction via risk-based decisions
Enables scalable real-time monitoring

🔮 Future Improvements
Concept drift detection
Automated retraining pipelines
Ensemble anomaly detection models
Real-time streaming (Kafka integration)

👤 Author
Laeba Jamil
Machine Learning Engineer
