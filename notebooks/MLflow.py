import mlflow
import pandas as pd
from mlflow.models import infer_signature

# Set our tracking server uri for logging
from mlflow.models.signature import ModelSignature
from numpy import ndarray
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params: dict[str, str | int] = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 42,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X=X_train, y=y_train)

# Predict on the test set
y_pred: ndarray = lr.predict(X=X_test)

# Calculate metrics
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
precision = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
recall = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
f1 = f1_score(y_true=y_test, y_pred=y_pred, average="macro")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")


# Set the tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment(experiment_name="MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params=params)

    # Log the loss metrics
    mlflow.log_metrics(
        metrics={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    # Set a tag that we can use to remind ourselves what the run was for
    mlflow.set_tag(
        key="Training Info", value="Basic Logistic Regression Model for Iris Data"
    )

    # Infer the Model Signature
    signature: ModelSignature = infer_signature(
        model_input=X_train, model_output=lr.predict(X=X_train)
    )

    # Log the Model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="iris_model-tracking-quickstart",
    )

    # Load the Model back for Predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris(return_X_y=False).feature_names

result = pd.DataFrame(data=X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])
