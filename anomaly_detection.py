import pandas as pd
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
file_path = "predictive_maintenance.csv"  # update path if needed
df = pd.read_csv(file_path)
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# 2️⃣ Select correct sensor features from AI4I dataset
features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

X = df[features].values

# 3️⃣ Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Train Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_scaled)

# Predict anomalies
anomaly_scores = -iso.score_samples(X_scaled)
anomaly_labels = (iso.predict(X_scaled) == -1).astype(int)

# Add results to dataframe
df['Anomaly'] = anomaly_labels
df['Anomaly_Score'] = anomaly_scores
print(df[['Anomaly', 'Anomaly_Score']].head(10))

# 5️⃣ Save anomalies to CSV
output_file = "ai4i2020_with_anomalies.csv"
df.to_csv(output_file, index=False)
print(f"✅ Anomaly results saved to {output_file}")

# 6️⃣ Visualize anomalies
plt.figure(figsize=(12, 5))
plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=anomaly_labels, cmap='coolwarm', s=20)
plt.title("Anomaly Scores in Industrial Sensor Data")
plt.xlabel("Sample Index")
plt.ylabel("Anomaly Score")
plt.show()

print("Anomaly counts:")
print(df['Anomaly'].value_counts())