import pandas as pd
import json
import numpy as np
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from .utils import get_data_for_test



docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters): 
    '''Deployment trigger config'''
    min_accuracy : float = 0.5
    

@step(enable_cache = False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data    
    
    
@step
def deployment_trigger(
    accuracy:float,
    config:DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the input model accuracy and decided if it is good enough to deploy or not."""
    return accuracy > config.min_accuracy
    
    
class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """"MLFlow deployment getter parameters
    
    Attributes:
        pipeline_name: name of the pipleline that deployed the MLFlow prediction
        step_name: The name of the step that deployed the MLflow prediction
        running: when this flaf is set, the step only returns a running service
        model_name: The name of the model that is deployed
    """
    pipeline_name:str
    step_name:str
    running:bool = True
    
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.
    
    Args:
        pipeline_name: Name of the pipeline that deployed the MLFlow prediction.
         server
        step_name: the name of the step that deployd the MLFlow prediction
            server
        running: when this flag is set, the step only return a running service
        model_name: the name of the model that is deployed
    """    
    
    # get the MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with the same pipeline name, step name, and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
        
    )
    
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}"
            f"step {pipeline_step_name} and model {model_name}"
            f"pipeline for the '{model_name}' model is currently"
            f"running."
    )
    return existing_services[0]

        
@step
def predictor(
    service:MLFlowDeploymentService,
    data: str
) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"],columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction
    
    
    

        
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path :str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )
         


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(pipeline_name:str, pipeline_step_name:str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False
    )
    prediction = predictor(service=service,data=data)
    return prediction