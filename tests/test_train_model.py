import pandas as pd
import pytest
from src.train_model import train_model
from unittest.mock import patch

df = pd.DataFrame({
    "Age": [22, 38, 26, 35, 28, 40, 19, 50],
    "Sex": [0, 1, 1, 0, 1, 0, 0, 1],
    "Pclass": [3, 1, 3, 2, 3, 1, 2, 1],
    "Survived": [0, 1, 1, 0, 1, 0, 0, 1],
})

# Test that the train_model function runs successfully and returns a model and metrics
def test_train_model_runs():
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    model, metrics = train_model(X, y, log_mlflow=False)

    assert model is not None
    assert isinstance(metrics, dict)
    assert "Accuracy" in metrics
    assert "F1-Score" in metrics

# Test that the trained model can make valid predictions on the input data
def test_model_can_predict():
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    model, _ = train_model(X, y, log_mlflow=False)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert all(p in [0, 1] for p in preds)


# Test that train_model raises a ValueError when given empty input data
def test_train_model_with_empty_data():
    X = pd.DataFrame(columns=["Age", "Sex", "Pclass"])
    y = pd.Series([], dtype=int)

    with pytest.raises(ValueError):
        train_model(X, y, log_mlflow=False)


# Test that train_model correctly calls MLflow logging functions when log_mlflow=True
# without needing a real MLflow server using mock functions
@patch("src.train_model.mlflow.xgboost.log_model")
@patch("src.train_model.mlflow.log_metric")
@patch("src.train_model.mlflow.log_params")
@patch("src.train_model.mlflow.start_run")
@patch("src.train_model.mlflow.set_tracking_uri")
def test_train_model_with_mlflow(
    mock_set_tracking_uri,
    mock_start_run,
    mock_log_params,
    mock_log_metric,
    mock_log_model
):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    model, metrics = train_model(X, y, log_mlflow=True)

    assert model is not None
    assert "Accuracy" in metrics

    # Verify MLflow logging calls
    mock_set_tracking_uri.assert_called_once()
    mock_start_run.assert_called_once()
    mock_log_params.assert_called()
    mock_log_metric.assert_called()
    mock_log_model.assert_called()


