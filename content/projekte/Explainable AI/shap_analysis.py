from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

# 1. Daten laden und vorbereiten
BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / 'penguins.csv')
data = data.dropna()  # Zeilen mit NaN-Werten entfernen (333 Datenpunkte verbleiben)

print("--- Datensatz-Info ---")
print(data.info())

target = "species"
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

X = data[features]
Y = data[target]

# Train-Test-Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.35,
    random_state=42,
    stratify=Y
)

# 2. Modell trainieren
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, Y_train)

# Klassische Feature Importance
print("\n--- Gini Feature Importance (Random Forest) ---")
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)
print(importance)

# Modell-Evaluation
print("\n--- Klassifikationsbericht ---")
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))

# 3. SHAP TreeExplainer initialisieren
print("\n--- SHAP Analyse wird berechnet ---")
explainer = shap.TreeExplainer(model)

# SHAP-Werte für das gesamte Testset berechnen
shap_values = explainer(X_test)
# shap_values shape: (Anzahl Samples, Anzahl Features, Anzahl Klassen)
class_names = list(model.classes_)
print(f"Klassen: {class_names}")

# 4. Globale Erklärung: Summary Bar Plot für alle Klassen
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, class_names=class_names, show=False)
plt.title("SHAP Global Feature Importance (alle Klassen)", fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# 5. Globale Erklärung: Beeswarm Plot für eine bestimmte Klasse (z.B. 'Gentoo')
target_class = "Gentoo"
target_class_idx = class_names.index(target_class)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values[:, :, target_class_idx], show=False)
plt.title(f"SHAP Beeswarm Plot — Einfluss der Features auf '{target_class}'", fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# 6. Lokale Erklärung: Einzelne Instanz erklären (z. B. Instanz 4)
instance_idx = 4
instance_to_explain = X_test.iloc[[instance_idx]]
true_label = Y_test.iloc[instance_idx]
predicted_label = model.predict(instance_to_explain)[0]
predicted_idx = class_names.index(predicted_label)
predicted_proba = model.predict_proba(instance_to_explain)[0]

print(f"\n--- Lokale Erklärung für Test-Instanz #{instance_idx} ---")
print("Feature-Werte:")
print(X_test.iloc[instance_idx])
print(f"Tatsächliche Art:  {true_label}")
print(f"Vorhergesagte Art: {predicted_label} (Wahrscheinlichkeit: {predicted_proba[predicted_idx]:.2%})")

# Waterfall Plot für die vorhergesagte Klasse der Instanz
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[instance_idx, :, predicted_idx], show=False)
plt.title(f"SHAP Waterfall Plot für Instanz #{instance_idx} (Vorhersage: '{predicted_label}')", fontsize=13, pad=15)
plt.tight_layout()
plt.show()
