import json
import mlflow
import os
import dagshub
from src.logger import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = "1d3fd59238b2289c912d55d35f5f1bbd99f6a14f"
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "seika-afk"
repo_name = "MLOPS-project2-"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info (run_id + model_path) from JSON."""
    with open(file_path, "r") as file:
        return json.load(file)


def register_model(model_name: str, model_info: dict):
    """
    Since DagsHub does not support MLflow Registry,
    just fetch the artifact (model.pkl) and log metadata.
    """
    try:
        run_id = model_info["run_id"]
        model_path = model_info["model_path"]  # should be 'models/model.pkl'

        model_uri = f"runs:/{run_id}/{model_path}"
        logging.info(f"Model URI: {model_uri}")

        # Log metadata about model (instead of registering)
        mlflow.log_dict(model_info, f"{model_name}_info.json")



    except Exception as e:
        logging.error("Error during model handling: %s", e)
        raise


def main():
    model_info_path = "reports/experiment_info.json"
    model_info = load_model_info(model_info_path)

    model_name = "my_model"
    register_model(model_name, model_info)


if __name__ == "__main__":
    main()

