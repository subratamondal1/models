# 1. Set the MLflow tracking server URI
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# 2. Create a new MLflow Experiment
mlflow.set_experiment(experiment_name="PyTorch Linear Regression")
# 3. Start an MLflow run
with mlflow.start_run():
    # 3.1 Log the hyperparameters
    pass
    # 3.2 Log the loss metrics
    # 3.3 Set a tag that we can use to remind ourselves what the run was for
    # 3.4 Infer the Model Signature
    # 3.5 Log the Model
# 4. Load the Model back for Predictions as a generic Python Function
