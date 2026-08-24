import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lime.lime_tabular
import sys
from pathlib import Path
import plotly.graph_objects as go

ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))
    
standard_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

def load_data():
    data = pd.read_csv(ORDNER/'penguins.csv')
    data = data.dropna() # removes rows with NaN values. 333 datapoints remain.
    return data

def train_blackbox(data, report=True):
    target = "species"
    features = standard_features
    X = data[features]
    Y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if report:
        print(classification_report(y_test, y_pred))
    return model, X_test, y_test

def create_lime_explainer(model, data, kernel_width,features=standard_features):
    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=data[features].values,
        feature_names=features,
        class_names=model.classes_,
        mode='classification',
        kernel_width=kernel_width
    )
    return explainer

def explain_instance(explainer, model, instance, num_features=4, num_samples=5000):
    # Define a prediction function for LIME
    def prediction_function(x):
        return model.predict_proba(x)

    # Generate explanation for the instance
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=prediction_function,
        num_features=num_features,
        num_samples=num_samples,
        top_labels=1
    )
    return explanation