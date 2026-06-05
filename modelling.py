import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = Path("dataset_preprocessing/breast_cancer_preprocessing.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("breast_cancer_preprocessing.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("SMSML_Najwan_Mursyidan")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="random_forest_autolog"):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1:", f1_score(y_test, y_pred, average="weighted"))
