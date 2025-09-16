import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import mlflow.models
import matplotlib.pyplot as plt
from xgboost import plot_importance

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
    print(f"Validation Accuracy: {acc:.4f}")

    # Log parameters, metrics, and model
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    # An example row from the dataset
    input_example = X_train.iloc[:1]
    mlflow.xgboost.log_model(model,  name="model",
                            input_example=input_example,
                            signature=mlflow.models.infer_signature(X_train, y_train))
