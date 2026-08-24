import pandas as pd

data = pd.read_csv('penguins.csv')
data = data.dropna() # removes rows with NaN values

print(data.info())

target = "species"
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

X = data[features]
Y = data[features]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.35,              # which proportion of data is used for testing
    random_state=42,            # seed for random split generator in case data has ordering
    stratify=Y                  # class labels are chosen from here AND represented proportionnaly in both sets
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    #bootstrap = False,
)

model.fit(X_train, Y_train)

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)

from sklearn.metrics import classification_report

Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
