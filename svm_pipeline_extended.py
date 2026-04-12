# ─────────────────────────────────────────────────────────────────────────────
# Extended Pipeline: SVM + Random Forest + Gradient Boosting Comparison
# Author : Pablo Soham | 2nd Year CS | 2026
# Dataset: UCI Smart Grid Stability (ID: 471)
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('smart_grid_stability.csv')
print(f"Dataset shape: {df.shape}")
print(df['stabf'].value_counts())

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
df = df.drop(columns=['stab'])
le = LabelEncoder()
df['stabf'] = le.fit_transform(df['stabf'])

X = df.drop(columns=['stabf'])
y = df['stabf']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── 3. Define Classifiers ────────────────────────────────────────────────────
classifiers = {
    'SVM (RBF)'       : SVC(kernel='rbf', C=10, gamma='scale',
                            probability=True, random_state=42),
    'Random Forest'   : RandomForestClassifier(n_estimators=100,
                            random_state=42, n_jobs=-1),
    'Gradient Boost'  : GradientBoostingClassifier(n_estimators=100,
                            random_state=42),
}

# ── 4. Train and Evaluate ────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Classifier Comparison ──────────────────────────────────────")
print(f"{'Classifier':<20} {'Accuracy':>10} {'AUC':>8} {'CV Mean':>10} {'CV Std':>8}")
print("-" * 62)

for name, clf in classifiers.items():
    print(f"Training {name}...", flush=True)
    clf.fit(X_train_sc, y_train)
    y_pred  = clf.predict(X_test_sc)
    y_proba = clf.predict_proba(X_test_sc)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(clf, X_train_sc, y_train,
                                cv=cv, scoring='accuracy')

    results[name] = {
        'clf'    : clf,
        'y_pred' : y_pred,
        'y_proba': y_proba,
        'acc'    : acc,
        'auc'    : auc,
        'cv_mean': cv_scores.mean(),
        'cv_std' : cv_scores.std(),
    }
    print(f"{name:<20} {acc:>10.4f} {auc:>8.4f} "
          f"{cv_scores.mean():>10.4f} {cv_scores.std():>8.4f}")

# ── 5. Detailed Report for SVM ───────────────────────────────────────────────
print("\n── Detailed Classification Report: SVM (RBF) ──────────────────")
print(classification_report(y_test, results['SVM (RBF)']['y_pred'],
      target_names=['unstable', 'stable']))

# ── 6. Comparison Bar Chart ──────────────────────────────────────────────────
os.makedirs('output_figures', exist_ok=True)
names = list(results.keys())
accs  = [results[n]['acc']     for n in names]
aucs  = [results[n]['auc']     for n in names]
cvs   = [results[n]['cv_mean'] for n in names]

x = np.arange(len(names))
w = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w, accs, w, label='Test Accuracy', color='#1A3C5E')
b2 = ax.bar(x,     aucs, w, label='AUC-ROC',       color='#2C5F8A')
b3 = ax.bar(x + w, cvs,  w, label='CV Accuracy',   color='#AED6F1')

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=8.5, color='#1A3C5E')

ax.set_ylim(0.85, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel('Score', fontsize=13, color='#1A3C5E')
ax.set_title('Figure 5: Classifier Comparison\n'
             'SVM (RBF) vs Random Forest vs Gradient Boosting',
             fontsize=12, color='#1A3C5E', pad=14)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('output_figures/fig5_classifier_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig5_classifier_comparison.png")

# ── 7. ROC Curves Comparison ─────────────────────────────────────────────────
colors_roc = ['#1A3C5E', '#1A6E3C', '#8B0000']
fig, ax = plt.subplots(figsize=(7, 6))

for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{name}  (AUC = {res['auc']:.4f})")

ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5,label='Random (AUC = 0.50)')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', fontsize=13, color='#1A3C5E')
ax.set_ylabel('True Positive Rate',  fontsize=13, color='#1A3C5E')
ax.set_title('Figure 6: ROC Curves — All Classifiers Compared',
             fontsize=12, color='#1A3C5E', pad=14)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('output_figures/fig6_roc_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_figures/fig6_roc_comparison.png")

# ── 8. Save Best Model ───────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['acc'])
joblib.dump({'model' : results[best_name]['clf'],
             'scaler': scaler,
             'name'  : best_name},
            'svm_grid_model_extended.pkl')
size_kb = os.path.getsize('svm_grid_model_extended.pkl') // 1024
print(f"\nBest model : {best_name}")
print(f"Model size : {size_kb} KB")
print("\nPipeline complete.")
