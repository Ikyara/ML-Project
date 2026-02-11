"""
Run this script ONCE to retrain the model on only the 6 features used in the Streamlit app.
Place student-por.csv and student-mat.csv in the same directory, then run:
    python train_model.py

It will generate:
    - random_forest_model.pkl
    - scaler.pkl
    - feature_columns.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ── 1. Load & merge data ────────────────────────────────────────────────────
por = pd.read_csv('./student-por.csv', sep=';')
mat = pd.read_csv('./student-mat.csv', sep=';')
df = pd.concat([por, mat], ignore_index=True)
print(f"Total records: {len(df)}")

# ── 2. Select ONLY the 6 features we use in the app ─────────────────────────
# This is the key fix: train the model on exactly the features the user controls.
# 'higher' is binary (yes/no), so we encode it as 1/0.
df['higher_yes'] = (df['higher'] == 'yes').astype(int)

FEATURES = ['G1', 'G2', 'absences', 'studytime', 'failures', 'higher_yes']
TARGET = 'G3'

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"Features used: {FEATURES}")
print(f"Feature shape: {X.shape}")

# ── 3. Train/test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 5. Train Random Forest ──────────────────────────────────────────────────
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
rf_pred = rf_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2 = r2_score(y_test, rf_pred)
print(f"\nRandom Forest Results:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# ── 7. Verify absences direction ─────────────────────────────────────────────
# Quick sanity check: predict with low vs high absences, all else equal
test_low = pd.DataFrame([[10, 10, 0, 2, 0, 1]], columns=FEATURES)
test_high = pd.DataFrame([[10, 10, 50, 2, 0, 1]], columns=FEATURES)
pred_low = rf_model.predict(scaler.transform(test_low))[0]
pred_high = rf_model.predict(scaler.transform(test_high))[0]
print(f"\nSanity check (G1=10, G2=10, studytime=2, failures=0, higher=yes):")
print(f"  Absences=0  → predicted G3: {pred_low:.2f}")
print(f"  Absences=50 → predicted G3: {pred_high:.2f}")
if pred_low >= pred_high:
    print("  ✓ More absences → lower/equal grade (correct direction)")
else:
    print("  ✗ WARNING: More absences → higher grade (unexpected)")

# ── 8. Feature importances ───────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\nFeature Importances:")
print(importance_df.to_string(index=False))

# ── 9. Save artifacts ───────────────────────────────────────────────────────
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

print("\n✓ Saved: random_forest_model.pkl, scaler.pkl, feature_columns.pkl")
print("Now run: streamlit run main.py")
