import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import mlflow.models
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Could store MLflow logs into a storage service such as s3 
#mlflow.set_tracking_uri("s3://my-bucket/mlflow")

# Load only the relevant columns
columns_to_use = ['Age', 'Sex', 'Pclass', 'Survived']
train_df = pd.read_csv("data/titanic_train.csv", usecols=columns_to_use)

# Drop rows where Age is missing
train_df = train_df.dropna(subset=['Age'])

# Encode categorical column (Sex)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

# Features and target
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# I will use the train csv to get a train / test split, since the test data is not labelled
# This is fine since this project is only for practise purposes

# Split train/test
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

# MLflow experiment tracking
mlflow.set_experiment("titanic_xgboost")

with mlflow.start_run():
    # Define model
    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.3,
        "eval_metric": "logloss",
        "random_state": 5
    }
    model = XGBClassifier(
        **params
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)

    # Evaluate
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Plot confusion matrix
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()

    # Save and log as MLflow artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close(fig_cm)  # Close figure to free memory



    # Log parameters, metrics, and model
    mlflow.log_params(params)
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1-Score", f1)

    fig, ax = plt.subplots()
    plot_importance(model, ax=ax)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close(fig)
    
    # An example row from the dataset
    input_example = X_train.iloc[:1]
    mlflow.xgboost.log_model(model,  name="model",
                            input_example=input_example,
                            signature=mlflow.models.infer_signature(X_train, y_train))
