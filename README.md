# 🕵️ Anomaly Detection Project

This repository contains implementations of anomaly detection using **Isolation Forest** and **XGBoost** in Python.  
The goal is to detect unusual data points (outliers) that do not follow the general pattern of the dataset.

---

## 📌 What is Anomaly Detection?
Anomaly detection refers to identifying rare items, events, or observations that differ significantly from the majority of the data.  
Such anomalies can indicate:
- Fraudulent transactions  
- Faulty equipment  
- Network intrusions  
- Unusual patterns in datasets  

---

## 🚀 Algorithms Implemented

### 1️⃣ Isolation Forest
- **Concept**:  
  Isolation Forest works on the principle of isolating anomalies instead of profiling normal data.  
  Anomalies are easier to isolate because they are fewer and different.  

- **Key Points**:  
  - Efficient for high-dimensional datasets.  
  - Unsupervised learning algorithm.  
  - Based on random partitioning of data.  



---

### 2️⃣ XGBoost for Anomaly Detection
- **Concept**:  
  XGBoost is a gradient boosting algorithm usually used for classification/regression.  
  For anomaly detection, it can be adapted by treating anomalies as a separate class in a supervised setting.  

- **Key Points**:  
  - Requires labeled dataset (normal vs anomaly).  
  - Extremely fast and accurate for large-scale data.  
  - Supports regularization to reduce overfitting.  


## ⚙️ Installation & Requirements

git clone https://github.com/patilyash948/Anomly-Detection.git
cd Anomly-Detection


```bash
pip install scikit-learn xgboost pandas numpy matplotlib

python "Isolation Forest.py"

python "XGBoost Anomaly Detection.py"


```bash
pip install scikit-learn xgboost pandas numpy matplotlib
