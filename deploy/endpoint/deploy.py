from typing import Optional, Dict, Sequence, Tuple
from google.cloud import aiplatform
from google.cloud.aiplatform import explain
import json

def create_endpoint_sample(
    project: str,
    display_name: str,
    location: str,
) -> aiplatform.Endpoint:
    """
    Creates a new endpoint in Vertex AI.

    Args:
        project (str): Google Cloud project ID.
        display_name (str): Display name for the new endpoint.
        location (str): Region where the endpoint will be located.

    Returns:
        aiplatform.Endpoint: Created endpoint object.
    """
    # Initialize Vertex AI client
    aiplatform.init(project=project, location=location)

    # Create endpoint with given display name
    endpoint = aiplatform.Endpoint.create(
        display_name=display_name,
        project=project,
        location=location,
    )

    # Print endpoint details
    print(f"Created endpoint: {endpoint.display_name}")
    print(f"Endpoint resource name: {endpoint.resource_name}")

    return endpoint

def deploy_model_with_dedicated_resources_sample(
    project: str,
    location: str,
    model_name: str,
    machine_type: str,
    endpoint: Optional[aiplatform.Endpoint] = None,
    deployed_model_display_name: Optional[str] = None,
    traffic_percentage: Optional[int] = 0,
    traffic_split: Optional[Dict[str, int]] = None,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None,
    explanation_parameters: Optional[explain.ExplanationParameters] = None,
    metadata: Optional[Sequence[Tuple[str, str]]] = (),
    sync: bool = True,
) -> aiplatform.Model:
    """
    Deploys a model to an endpoint with dedicated resources in Vertex AI.

    Args:
        project (str): Google Cloud project ID.
        location (str): Region where the model and endpoint are located.
        model_name (str): Full resource name of the model to deploy.
        machine_type (str): Machine type to use for the deployment.
        endpoint (aiplatform.Endpoint, optional): Existing endpoint to deploy the model to.
        deployed_model_display_name (str, optional): Display name for the deployed model.
        traffic_percentage (int, optional): Percentage of traffic to send to this model version.
        traffic_split (Dict[str, int], optional): Traffic split configuration between model versions.
        min_replica_count (int): Minimum number of replicas to deploy.
        max_replica_count (int): Maximum number of replicas to deploy.
        accelerator_type (str, optional): Type of hardware accelerator (e.g., "NVIDIA_TESLA_K80").
        accelerator_count (int, optional): Number of hardware accelerators to attach.
        explanation_metadata (explain.ExplanationMetadata, optional): Metadata for explaining predictions.
        explanation_parameters (explain.ExplanationParameters, optional): Parameters for explaining predictions.
        metadata (Sequence[Tuple[str, str]], optional): Additional metadata tags for the model deployment.
        sync (bool, optional): Whether to wait for the deployment operation to complete.

    Returns:
        aiplatform.Model: Deployed model object.
    """
    # Initialize Vertex AI client
    aiplatform.init(project=project, location=location)
    
    models = aiplatform.Model.list()
    for model in models:
        if model.display_name == model_name:
            model_id = model.name

    # Create model instance with specified model name
    model = aiplatform.Model(model_name=model_id)

    # Deploy model to endpoint with dedicated resources
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=traffic_percentage,
        traffic_split=traffic_split,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        metadata=metadata,
        sync=sync,
    )

    # Wait for the deployment operation to complete
    model.wait()

    # Print model details
    print(f"Deployed model: {model.display_name}")
    print(f"Model resource name: {model.resource_name}")

    return model

if __name__ == "__main__":
    
    file_path = 'config.json'

    with open(file_path, 'r') as file:
        config = json.load(file)

    PROJECT_ID = config.get('PROJECT_ID', 'beaming-signal-428023-h8')
    LOCATION = config.get('LOCATION', 'us-central1')
    MODEL_ID = config['PIPELINE_PARAMS']['model_display_name']

    # Create an endpoint
    endpoint = create_endpoint_sample(PROJECT_ID, "My Endpoint", LOCATION)


    model_name = MODEL_ID
    machine_type = "n1-standard-4"

    deploy_model_with_dedicated_resources_sample(
        PROJECT_ID, LOCATION, model_name, machine_type, endpoint=endpoint
    )
