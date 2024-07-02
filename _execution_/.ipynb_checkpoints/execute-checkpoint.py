import json
import sys
from google.cloud import aiplatform

def envar_get_data(config: dict):
    """
    Extracts environment-specific data from the configuration dictionary.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        pipeline details: project_id, description, display_name, region, pipeline_params, service_account
    """
    
    project_id = config.get('PIPELINE_PROJECT_ID')
    description = config.get('PIPELINE_DESCRIPTION')
    display_name = config.get('PIPELINE_DISPLAY_NAME')
    region = config.get('LOCATION')
    pipeline_params = config.get('PIPELINE_PARAMS')
    service_account = config.get('SERVICE_ACCOUNT')

    return project_id, description, display_name, region, pipeline_params, service_account


def main(job_id: str = None,
         enable_caching: bool = False,
         monitoring: bool = False,
         config_path: str = None):
    """
    Main function to initialize AI Platform, generate and submit a pipeline job.

    Args:
        pipeline_name (str): The name of the pipeline.
        job_id (str, optional): The job ID. Defaults to None.
        enable_caching (bool, optional): Whether to enable caching. Defaults to False.
        monitoring (bool, optional): Whether to enable monitoring. Defaults to False.
        config_path (str, optional): The path to the configuration file. Defaults to None.
    """

    # Load configuration from file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        print(f'An error occurred during config file loading')
        sys.exit(1)

    # Extract environment data from configuration
    try:
        project_id, description, display_name, region, pipeline_params, service_account = envar_get_data(config)

        # Initialize AI Platform
        print('Initializing AI Platform...')
        aiplatform.init(project=project_id, location=region)

        # Generate pipeline job
        print('Generating pipeline job...')
        job = aiplatform.pipeline_jobs.PipelineJob(
            display_name=display_name,
            template_path="_execution_/compiled_pipeline.json", 
            job_id=job_id,
            enable_caching=enable_caching,
            project=project_id,
            location=region,
            failure_policy='slow',
            parameter_values=pipeline_params
        )

        # Submit pipeline job
        print('Submitting pipeline job...')
        job.submit(service_account=service_account)

        print('Pipeline job submitted successfully.')

    except Exception as e:
        print(e)
        print('An error occurred during pipeline job initialization or submission')
        sys.exit(1)

if __name__ == '__main__':
    
    main(
        config_path='config.json'
    )
