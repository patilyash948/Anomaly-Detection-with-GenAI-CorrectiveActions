import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
file_path = "predictive_maintenance.csv"  # Update path if needed
df = pd.read_csv(file_path)

# 2️⃣ Clean column names (remove spaces)
df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# 3️⃣ Select sensor features
features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# 4️⃣ Identify target column
# Try common names for AI4I dataset
possible_targets = ['Machine failure', 'Machine Failure', 'failure', 'Failure', 'Target']
target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("❌ No valid target column found! Please check your CSV.")

print(f"✅ Using target column: {target_column}")

X = df[features].values
y = df[target_column].values

# 5️⃣ Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y 
)

# 7️⃣ Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    scale_pos_weight=int(sum(y==0)/sum(y==1))  #handle class imbalance 
)
xgb_model.fit(X_train, y_train) 

# 8️⃣ Make predictions
y_pred = xgb_model.predict(X_test)

# 9️⃣ Evaluate
print("🔹 Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report")
print(classification_report(y_test, y_pred))

# 🔟 Add predictions and probabilities to original dataframe
df['Predicted_Failure'] = xgb_model.predict(X_scaled)
df['Failure_Probability'] = xgb_model.predict_proba(X_scaled)[:, 1]

print(df[['Predicted_Failure', 'Failure_Probability']].head(10)) 

# 1️⃣1️⃣ Save predictions to CSV
output_file = "ai4i2020_with_predictions.csv"
df.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}")

# 1️⃣2️⃣ Visualize predicted failure probability
plt.figure(figsize=(12, 5))
plt.scatter(range(len(df)), df['Failure_Probability'], c=df['Predicted_Failure'], cmap='coolwarm', s=20)
plt.title("Predicted Failure Probabilities")
plt.xlabel("Sample Index")
plt.ylabel("Failure Probability")
plt.show()
