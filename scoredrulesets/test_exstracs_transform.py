"""Test: Vergleiche native ExSTraCS-Prediction mit transformierter ScoredRuleSet-Prediction"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from skExSTraCS import ExSTraCS
import numpy as np

# Daten laden
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ExSTraCS trainieren
est = ExSTraCS(learning_iterations=5000, N=1000, random_state=42)
est.fit(X_train, y_train)

# Native Prediction
y_native = est.predict(X_test)
f1_native = f1_score(y_test, y_native, average='weighted')
print(f"F1 nativ: {f1_native:.4f}")

# Transformation zu ScoredRuleSet
from scoredrulesets.estimators.ruleset_transform import exstracs_to_scored_ruleset
feature_names = [f"f{i}" for i in range(X.shape[1])]
class_labels = [int(c) for c in np.unique(y_train)]

ruleset = exstracs_to_scored_ruleset(est, class_labels, feature_names)
print(f"Regeln: {len(ruleset.rules)}")
print(f"Atome gesamt: {sum(len(r.atoms) for r in ruleset.rules)}")

# Prüfe: Wie viele Regeln haben keine Atome?
n_no_atoms = sum(1 for r in ruleset.rules if len(r.atoms) == 0)
print(f"Regeln ohne Atome: {n_no_atoms}")

# Prediction via ScoredRuleSet
from scoredrulesets.runtime import predict
y_transformed = predict(ruleset, X_test)
print(f"y_transformed dtype={y_transformed.dtype}, unique={np.unique(y_transformed)}, first5={y_transformed[:5]}")
print(f"y_test dtype={y_test.dtype}, unique={np.unique(y_test)}, first5={y_test[:5]}")
# Konvertiere y_transformed zu int falls nötig
y_transformed_int = np.array([int(v) for v in y_transformed])
f1_transformed = f1_score(y_test, y_transformed_int, average='weighted')
print(f"F1 transformiert: {f1_transformed:.4f}")

# Vergleich pro Sample
mismatches = 0
for i in range(len(y_test)):
    native_val = y_native[i]
    trans_val = y_transformed[i]
    if native_val != trans_val:
        mismatches += 1
        if mismatches <= 10:
            print(f"  Mismatch Sample {i}: nativ={native_val} transformiert={trans_val} true={y_test[i]}")
print(f"Mismatches: {mismatches}/{len(y_test)}")
print(f"F1 nativ: {f1_native:.4f}  |  F1 transformiert: {f1_transformed:.4f}")



