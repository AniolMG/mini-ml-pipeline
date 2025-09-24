import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting...")

# -------------------------------
# Core training function
# -------------------------------
def train_model(X, y, log_mlflow=True, mlflow_params=None):
    """
    Train XGBoost classifier and optionally log to MLflow.

    Returns:
        model: trained XGBClassifier
        metrics: dict with Accuracy and F1-Score
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=5, stratify=y
    )

    # Default parameters
    params = mlflow_params or {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.3,
        "eval_metric": "logloss",
        "random_state": 5
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    metrics = {"Accuracy": acc, "F1-Score": f1}

    if log_mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("titanic_s3")
        MODEL_NAME = "TitanicModel"

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("F1-Score", f1)

            # Optionally log plots
            cm = confusion_matrix(y_val, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            plt.close(fig_cm)

            signature = infer_signature(X_train, y_train)
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                input_example=X_train.iloc[:1],
                signature=signature
            )

    return model, metrics


# -------------------------------
# If run as script
# -------------------------------
if __name__ == "__main__":
    # Load & preprocess
    columns_to_use = ['Age', 'Sex', 'Pclass', 'Survived']
    train_df = pd.read_csv("../data/titanic_train.csv", usecols=columns_to_use)
    train_df = train_df.dropna(subset=['Age'])
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']

    model, metrics = train_model(X, y, log_mlflow=True)
    print("Training completed. Metrics:", metrics)
