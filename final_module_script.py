import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

import matplotlib.pyplot as plt



CSV_PATH = r"C:\Users\afaan\noon_only_2010-2025.csv" 
df = pd.read_csv(CSV_PATH)

df["datetime"] = pd.to_datetime(df["Time (UTC)"], format="%d-%b-%y %H:%M:%S")
df = df[df["datetime"].dt.hour == 12].copy()


df = df.dropna(subset=[
    "Significant wave height",
    "Wave period",
    "Wave from direction",
    "North surface currents",
    "East surface currents",
])


df["month"] = df["datetime"].dt.month
df["day_of_year"] = df["datetime"].dt.dayofyear


radians = np.deg2rad(df["Wave from direction"])
df["wave_dir_sin"] = np.sin(radians)
df["wave_dir_cos"] = np.cos(radians)

# =========================
# 5) Construct tomorrow's target label
# =========================
df = df.sort_values("datetime").reset_index(drop=True)

df["is_choppy_today"] = (df["Significant wave height"] >= 0.3).astype(int)
df["tomorrow_is_choppy"] = df["is_choppy_today"].shift(-1)

# For analysis only (not a feature)
df["tomorrow_wave_height"] = df["Significant wave height"].shift(-1)

df = df.dropna(subset=["tomorrow_is_choppy"]).copy()
df["tomorrow_is_choppy"] = df["tomorrow_is_choppy"].astype(int)



# =========================
# 6) Select features (today only)
# =========================
feature_cols = [
    "Temperature",
    "Salinity",
    "Significant wave height",
    "Wave period",
    "wave_dir_sin",
    "wave_dir_cos",
    "North surface currents",
    "East surface currents",
    "month",
    "day_of_year",
]

X = df[feature_cols].copy()
y = df["tomorrow_is_choppy"].copy()

# =========================
# 7) Train/test split, then train/val split (for threshold tuning)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,        # 0.25 of 0.80 = 0.20 of total, so you get 60/20/20
    random_state=42,
    stratify=y_train
)

# =========================
# 8) Baseline: always predict majority class
# =========================
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train2, y_train2)
y_dummy_test = dummy.predict(X_test)

print("\n===== DUMMY BASELINE (Always majority class) =====")
print("Test accuracy:", accuracy_score(y_test, y_dummy_test))
print("Confusion matrix:\n", confusion_matrix(y_test, y_dummy_test))
print(classification_report(y_test, y_dummy_test, digits=3))

# =========================
# 9) Model: Random Forest 
# =========================
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train2, y_train2)

# Default threshold (0.50) predictions
y_train_pred = clf.predict(X_train2)
y_test_pred_default = clf.predict(X_test)

print("\n===== RANDOM FOREST (Default 0.50 threshold) =====")
print("Train accuracy:", accuracy_score(y_train2, y_train_pred))
print("Test accuracy:", accuracy_score(y_test, y_test_pred_default))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred_default))
print(classification_report(y_test, y_test_pred_default, digits=3))

# =========================
# 10) Threshold tuning to increase recall (use VALIDATION set only)
# =========================
target_recall = 0.70  # set this to whatever you want to argue for (0.60, 0.75, 0.80, etc.)

p_val = clf.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, p_val)

pr = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1],
})

# Choose: among thresholds that hit the recall target, pick the HIGHEST threshold
# (highest threshold = fewer false alarms while still meeting recall target)
eligible = pr[pr["recall"] >= target_recall].sort_values("threshold", ascending=False)

if eligible.empty:
    # If you can't reach the target recall, pick the threshold that maximizes recall
    best_row = pr.sort_values(["recall", "precision"], ascending=False).head(1)
    print("\nCould not reach target recall on validation. Using best recall threshold instead.")
else:
    best_row = eligible.head(1)

best_t = float(best_row["threshold"].iloc[0])

print("\n===== THRESHOLD SELECTION (Validation only) =====")
print("Target recall:", target_recall)
print("Chosen threshold:", best_t)
print(best_row)

# Apply tuned threshold on test
p_test = clf.predict_proba(X_test)[:, 1]
y_test_pred_tuned = (p_test >= best_t).astype(int)

print("\n===== RANDOM FOREST (Tuned threshold on TEST) =====")
print("Test accuracy:", accuracy_score(y_test, y_test_pred_tuned))
cm_tuned = confusion_matrix(y_test, y_test_pred_tuned)
print("Confusion matrix:\n", cm_tuned)
print(classification_report(y_test, y_test_pred_tuned, digits=3))

# =========================
# 11) Feature importances (same)
# =========================
fi = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
print("\nFeature importances:")
print(fi)

# =========================
# 12) Misclassified samples (using tuned threshold)
# =========================
test_df = df.loc[X_test.index].copy()
test_df["true_label"] = y_test
test_df["pred_label"] = y_test_pred_tuned

mis_df = test_df[test_df["true_label"] != test_df["pred_label"]].copy()
print("\nNumber of misclassified test samples:", len(mis_df))

mis_inspect_cols = [
    "datetime",
    "Temperature",
    "Salinity",
    "Significant wave height",
    "Wave period",
    "Wave from direction",
    "North surface currents",
    "East surface currents",
    "month",
    "day_of_year",
    "tomorrow_wave_height",
    "tomorrow_is_choppy",
    "true_label",
    "pred_label",
]

mis_df_sorted = mis_df.sort_values("datetime")[mis_inspect_cols]
print("\nFirst 5 misclassified cases (tuned threshold):")
print(mis_df_sorted.head(5))

# =========================
# 13) Confusion matrix heatmap (tuned)
# =========================
fig, ax = plt.subplots()
im = ax.imshow(cm_tuned, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Calm (0)", "Choppy (1)"])
ax.set_yticklabels(["Calm (0)", "Choppy (1)"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Tuned threshold = {best_t:.3f})")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_tuned[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

# =========================
# 14) Feature importance bar chart (same)
# =========================
fi_sorted = fi.sort_values("importance", ascending=True)
plt.figure(figsize=(6, 4))
plt.barh(fi_sorted["feature"], fi_sorted["importance"])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()
