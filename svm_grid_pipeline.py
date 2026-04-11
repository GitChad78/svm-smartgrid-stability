# ============================================================
# SVM-Based Smart Grid Stability Classification
# Applied Machine Learning — Research Paper
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc)
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Dataset ---
df = pd.read_csv("data/smart_grid_stability.csv")
print("Shape:", df.shape)
print(df.head())
print(df['stabf'].value_counts())

# --- 2. Preprocess ---
X = df.drop(columns=['stab', 'stabf'])
y = (df['stabf'] == 'unstable').astype(int)   # 1=unstable, 0=stable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --- 3. Train SVM (RBF kernel) ---
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
y_pred = svm.predict(X_test_sc)

# --- 4. Metrics ---
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Stable','Unstable']))

# Save metrics to text
with open("outputs/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(classification_report(y_test, y_pred, target_names=['Stable','Unstable']))

# --- 5. Confusion Matrix Plot ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stable','Unstable'],
            yticklabels=['Stable','Unstable'])
plt.title('Confusion Matrix — SVM (RBF Kernel)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=300)
plt.close()

# --- 6. ROC Curve ---
y_prob = svm.predict_proba(X_test_sc)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — SVM Smart Grid Classifier')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=300)
plt.close()

# --- 7. Feature Importance (via permutation proxy) ---
from sklearn.inspection import permutation_importance
result = permutation_importance(svm, X_test_sc, y_test, n_repeats=10, random_state=42)
feat_imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8,6))
feat_imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Permutation Method')
plt.xlabel('Mean Accuracy Decrease')
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300)
plt.close()

# --- 8. Class Distribution ---
plt.figure(figsize=(5,4))
df['stabf'].value_counts().plot(kind='bar', color=['#2ecc71','#e74c3c'], edgecolor='black')
plt.title('Class Distribution in Dataset')
plt.xlabel('Grid State'); plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=300)
plt.close()

# --- 9. Cross-Validation ---
cv_scores = cross_val_score(svm, X_train_sc, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
with open("outputs/metrics.txt", "a") as f:
    f.write(f"\n5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# --- 10. Save Model ---
joblib.dump(svm, "models/svm_grid_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nAll plots saved to /plots/  |  Model saved to /models/")
