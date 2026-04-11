# ─────────────────────────────────────────────────────────────
# SVM-Based Smart Grid Stability Prediction Pipeline
# Author: Pablo Soham | 2nd Year CS | 2026
# Dataset: UCI Smart Grid Stability (ID: 471)
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv('smart_grid_stability.csv')
print("Dataset shape:", df.shape)
print(df['stabf'].value_counts())

# ── 2. Preprocessing ─────────────────────────────────────────
df = df.drop(columns=['stab'])           # remove to prevent leakage
le = LabelEncoder()
df['stabf'] = le.fit_transform(df['stabf'])   # stable=1, unstable=0

X = df.drop(columns=['stabf'])
y = df['stabf']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. Train SVM ─────────────────────────────────────────────
model = SVC(kernel='rbf', C=10, gamma='scale',
            probability=True, random_state=42)
model.fit(X_train_sc, y_train)
print("Model trained successfully.")

# ── 4. Evaluate ──────────────────────────────────────────────
y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred,
      target_names=['unstable', 'stable']))
print("Test Accuracy :", accuracy_score(y_test, y_pred))
print("AUC-ROC       :", roc_auc_score(y_test, y_proba))

# ── 5. Cross-Validation ──────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_sc, y_train,
                             cv=cv, scoring='accuracy')
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Unstable','Stable'],
            yticklabels=['Unstable','Stable'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("Saved: confusion_matrix.png")

# ── 7. ROC Curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()
print("Saved: roc_curve.png")

# ── 8. Feature Importance ────────────────────────────────────
perm = permutation_importance(model, X_test_sc, y_test,
                              n_repeats=10, random_state=42,
                              scoring='accuracy')
sorted_idx = perm.importances_mean.argsort()
plt.figure(figsize=(8, 6))
plt.barh(X.columns[sorted_idx], perm.importances_mean[sorted_idx])
plt.xlabel('Mean Accuracy Decrease')
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("Saved: feature_importance.png")

# ── 9. Kernel Comparison ─────────────────────────────────────
print("\n── Kernel Comparison ──")
for k in ['rbf', 'linear', 'poly', 'sigmoid']:
    m = SVC(kernel=k, C=10, gamma='scale', random_state=42)
    m.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_sc))
    print(f"  Kernel={k:<8}: {acc:.4f}")

# ── 10. Save Model ───────────────────────────────────────────
joblib.dump({'model': model, 'scaler': scaler}, 'svm_grid_model.pkl')
size_kb = os.path.getsize('svm_grid_model.pkl') // 1024
print(f"\nModel saved: svm_grid_model.pkl ({size_kb} KB)")
print("\n✅ Pipeline complete.")
