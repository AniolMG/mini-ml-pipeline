import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

print("Starting...")


# -------------------------------
# Configure MLflow server
# -------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow server
print("URI set!")
mlflow.set_experiment("titanic_s3") # MLflow experiment         
print("Experiment set!")
MODEL_NAME = "TitanicModel"  # Name in Model Registry

# -------------------------------
# Load and preprocess data
# -------------------------------
columns_to_use = ['Age', 'Sex', 'Pclass', 'Survived']
train_df = pd.read_csv("../data/titanic_train.csv", usecols=columns_to_use)
train_df = train_df.dropna(subset=['Age'])
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

print("Train test split finished!")

# -------------------------------
# Start MLflow run
# -------------------------------
with mlflow.start_run() as run:
    # Model parameters
    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.3,
        "eval_metric": "logloss",
        "random_state": 5
    }

    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    print("Model fitted!")

    # Predictions and metrics
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print("Metrics computed!")

    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1-Score", f1)

    print("Metrics logged!")

    # -------------------------------
    # Log artifacts (plots)
    # -------------------------------
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()
    #plt.savefig("confusion_matrix.png")
    #mlflow.log_artifact("confusion_matrix.png")
    plt.close(fig_cm)

    # Feature importance
    fig, ax = plt.subplots()
    plot_importance(model, ax=ax)
    plt.tight_layout()
    #plt.savefig("feature_importance.png")
    #mlflow.log_artifact("feature_importance.png")
    plt.close(fig)
    print("Plotting finished!")

    # -------------------------------
    # Log model to MLflow server + Model Registry
    # -------------------------------
    input_example = X_train.iloc[:1]
    signature = infer_signature(X_train, y_train)

    # log_model automatically stores to server artifact root
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=input_example,
        signature=signature
    )
    print("Model logged!")
    
    print(f"Run completed. Model registered in MLflow Model Registry as '{MODEL_NAME}'.")

    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    run_info = client.get_run(run_id)
    print(run_info.info.artifact_uri)

