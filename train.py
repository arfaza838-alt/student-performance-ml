import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# DEMO DATA ONLY:
# These are illustrative records, not real student data.
data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
        4, 5, 6, 7, 8, 1, 3, 5, 7, 9,
        2, 4, 6, 8, 9, 1, 4, 6, 7, 9
    ],
    "attendance": [
        45, 50, 55, 60, 65, 70, 80, 90, 40, 60,
        70, 75, 85, 90, 95, 35, 50, 70, 80, 95,
        45, 65, 75, 85, 98, 40, 60, 80, 88, 99
    ],
    "previous_score": [
        30, 40, 45, 50, 55, 60, 70, 80, 35, 45,
        50, 60, 65, 75, 85, 25, 40, 55, 70, 90,
        35, 50, 65, 80, 90, 20, 45, 70, 75, 95
    ],
    "passed": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
        0, 0, 1, 1, 1, 0, 0, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_score"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Demo model accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model for the prediction application.
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")

# Test one example.
sample = pd.DataFrame(
    [[5, 75, 60]],
    columns=["study_hours", "attendance", "previous_score"]
)

result = model.predict(sample)[0]
print("Demo prediction:", "Pass" if result == 1 else "Needs support")