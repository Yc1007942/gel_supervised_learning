import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

def extract_flow_features(flow):
    Ox, Oy, Cx, Cy, Occ = flow
    dx = np.array(Cx) - np.array(Ox)
    dy = np.array(Cy) - np.array(Oy)
    mag = np.hypot(dx, dy)
    return {
        'dx_mean': dx.mean(),
        'dx_std':  dx.std(),
        'dy_mean': dy.mean(),
        'dy_std':  dy.std(),
        'mag_mean': mag.mean(),
        'mag_std':  mag.std(),
        'occ_ratio': np.mean(np.array(Occ) < 0)
    }

# ─── 1) Load all flow.json into a DataFrame ────────────────────────────────────────
rows = []
for mat_dir in sorted(glob.glob("material_*")):
    mat_id = int(mat_dir.split("_")[-1])
    for cycle_dir in sorted(glob.glob(os.path.join(mat_dir, "cycle_*"))):
        path = os.path.join(cycle_dir, "flow.json")
        if not os.path.isfile(path): 
            continue
        flow = json.load(open(path))
        feats = extract_flow_features(flow)
        feats['material'] = mat_id
        rows.append(feats)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} samples across {df['material'].nunique()} materials.")

# ─── 2) Train/Test split + pipeline ────────────────────────────────────────────────
X = df.drop('material', axis=1)
y = df['material']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(n_estimators=100, random_state=42))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
acc = pipe.score(X_test, y_test)
print(f"Test accuracy: {acc*100:.1f}%")

# ─── 3) Scatter plot (mag_mean vs mag_std) ─────────────────────────────────────────
plt.figure()
for mat in sorted(df['material'].unique()):
    sub = df[df['material'] == mat]
    plt.scatter(sub['mag_mean'], sub['mag_std'], label=f"Mat {mat}")
plt.xlabel('Magnitude Mean')
plt.ylabel('Magnitude Std')
plt.title('Scatter: mag_mean vs mag_std')
plt.legend()
plt.tight_layout()
plt.show()

# ─── 4) Correlation matrix (table + heatmap) ───────────────────────────────────────
corr = df.corr()
print("\nFeature correlation matrix:")
print(corr.round(2))

plt.figure()
plt.imshow(corr.values, interpolation='nearest')
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Matrix Heatmap')
plt.colorbar()
plt.tight_layout()
plt.show()

# ─── 5) Feature importances ────────────────────────────────────────────────────────
importances = pipe.named_steps['clf'].feature_importances_
feat_names  = X.columns
plt.figure()
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feat_names, rotation=90)
plt.title('Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# ─── 6) Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.xticks(range(len(cm)), sorted(y.unique()))
plt.yticks(range(len(cm)), sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.colorbar()
plt.tight_layout()
plt.show()
