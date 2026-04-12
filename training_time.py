import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('smart_grid_stability.csv').drop(columns=['stab'])
df['stabf'] = LabelEncoder().fit_transform(df['stabf'])
X = df.drop(columns=['stabf'])
y = df['stabf']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
classifiers = {
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale',
                     probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100,
                     random_state=42, n_jobs=-1),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100,
                     random_state=42),
}
print("\n── Training Time Comparison ──────────────────")
print(f"{'Classifier':<20} {'Train Time':>12} {'Inference':>12}")
print("-" * 46)
for name, clf in classifiers.items():
    t0 = time.time()
    clf.fit(X_train_sc, y_train)
    train_t = time.time() - t0
    t0 = time.time()
    for _ in range(1000):
        clf.predict(X_test_sc[:1])
    infer_t = (time.time() - t0) / 1000 * 1000
    print(f"{name:<20} {train_t:>10.2f}s {infer_t:>10.4f}ms")
print("\nDone.")
