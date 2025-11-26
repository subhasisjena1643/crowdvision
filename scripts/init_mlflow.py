"""
Initialize MLflow experiment tracking
"""

import mlflow
from pathlib import Path
import os
from dotenv import load_dotenv


def init_mlflow():
    """Initialize MLflow tracking"""
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "crowdvision")
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": "crowdvision",
                    "description": "AI/ML models for crowd safety monitoring"
                }
            )
            print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        # Set active experiment
        mlflow.set_experiment(experiment_name)
        
        # Create sub-experiments for different model types
        model_types = [
            "density_estimation",
            "spatiotemporal_forecasting",
            "anomaly_detection",
            "person_reid",
            "resource_allocation"
        ]
        
        for model_type in model_types:
            sub_experiment_name = f"{experiment_name}_{model_type}"
            sub_experiment = mlflow.get_experiment_by_name(sub_experiment_name)
            if sub_experiment is None:
                sub_experiment_id = mlflow.create_experiment(
                    sub_experiment_name,
                    tags={
                        "project": "crowdvision",
                        "model_type": model_type
                    }
                )
                print(f"Created sub-experiment: {sub_experiment_name}")
        
        print("\nMLflow initialization complete!")
        print(f"Access MLflow UI at: {tracking_uri}")
        
    except Exception as e:
        print(f"Error initializing MLflow: {e}")
        print("Make sure MLflow server is running:")
        print("  mlflow server --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    init_mlflow()
