# 🏦 Bank Customer Churn Prediction

A machine learning project to predict whether a bank customer will leave or stay,
based on demographic and account information.




---

## Overview

Banks lose significant revenue when customers close their accounts.
This project builds a Random Forest classification model that identifies
high-risk customers before they churn, enabling the bank to take
proactive retention actions.

**Input:** Customer data in CSV format  
**Output:** Churn probability (0–1) per customer  
**Model:** Random Forest with ROC-AUC of ~0.66

---

## Dataset

| Feature | Description | Type |
|---------|-------------|------|
| `Age` | Customer age | Numeric |
| `Balance` | Account balance | Numeric |
| `EstimatedSalary` | Estimated annual salary | Numeric |
| `NumOfProducts` | Number of bank products held | Numeric |
| `IsActiveMember` | Active customer status | Binary (0/1) |
| `Gender` | Gender | Binary (0/1) (Male=1 , Female=0) |
| `Exited` | Churned? **(Target)** | Binary (0/1) |

- **Records:*. 50 customers  
- **Churn rate:** ~20% (imbalanced — handled with SMOTE)  
- **Source format:** CSV

---

## Project Structure

```
bank-churn-prediction/
│
├── data/
│   └── bank_churn.csv
│
│
├── app/
│   └── Bank_Prediction.ipynb
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```
CSV Data
    ↓
Data Validation (null check, type check, duplicates)
    ↓
Exploratory Data Analysis
    ↓
Preprocessing (Encoding → Scaling → Train/Test Split → SMOTE)
    ↓
Model Training (Random Forest)
    ↓
Evaluation (ROC-AUC · F1 · Confusion Matrix)
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Streamlit Dashboard
```

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.66 |
| Recall  | 0.33 |
| Accuracy | 0.80 |
| Precision | 1.0 |

**Key findings:**
- Customers older than 50 have the highest churn probability
- Inactive members are 2x more likely to churn
- Customers with 3–4 products show unexpectedly high churn rates
- `Age` and `IsActiveMember` are the most important features

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python src/train.py
```

**4. Run the dashboard**
```bash
streamlit run app/streamlit_app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
Random Forest 
matplotlib


---

## Roadmap

- [ ] Add tenure and transaction history features
- [ ] Compare with XGBoost and other models
- [ ] Deploy as REST API with FastAPI
- [ ] Add automated alerts for high-risk customers

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
