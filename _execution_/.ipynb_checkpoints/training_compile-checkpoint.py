#!/usr/bin/env python
# coding: utf-8

# ## Installing dependencies

# In[37]:


import os, sys
import json
from datetime import datetime
import google.cloud.aiplatform as aiplatform
from typing import List
from typing import NamedTuple
import kfp
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import (component, Input, Model, Output, Dataset, 
                        Artifact, OutputPath, ClassificationMetrics, 
                        Metrics, InputPath)


# ## Defining variables

# In[2]:


file_path = 'config.json'

with open(file_path, 'r') as file:
    config = json.load(file)
    
PIPELINE_DISPLAY_NAME = config.get('PIPELINE_DISPLAY_NAME', 'my_pipeline')
PIPELINE_DESCRIPTION = config.get('PIPELINE_DESCRIPTION', 'pipeline challenge')


# ## Creating components

# In[38]:


@dsl.component()
def error_raising(msg: str) -> None:
    raise(msg)


# In[39]:


@dsl.component(packages_to_install=['pandas', 'google-cloud-bigquery'])
def load_from_bq(
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str,
    df_data: Output[Dataset]
) -> None:
    """
    Load data from a specified BigQuery table and return a pandas DataFrame.
    
    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The dataset ID within BigQuery.
        table_id (str): The table ID within the dataset.
        location (str): The location of the BigQuery dataset. Default is "EU".
    
    Returns:
        pd.DataFrame: DataFrame containing the data from the BigQuery table.
    """
    
    from google.cloud import bigquery
    
    try:
        # Initiate the BigQuery client to connect with the project.
        bq_client = bigquery.Client(project=project_id, location=location)
        
        # Load data from the BigQuery table.
        dataset_ref = bq_client.dataset(dataset_id, project=project_id)
        table_ref = dataset_ref.table(table_id)
        table = bq_client.get_table(table_ref)
        rows = bq_client.list_rows(table)

        # Convert to a pandas DataFrame.
        df_loaded = rows.to_dataframe()

        if not df_loaded.empty:
            df_loaded.to_csv(df_data.path, index=False)
        else:
            raise ValueError("Table content is empty.")
    
    except Exception as e:
        print(f"An error occurred during table load: {e}")
        raise


# In[40]:


@dsl.component(packages_to_install=['pandas', 'google-cloud-storage'])
def load_from_gcs(
    bucket_name: str,
    file_path: str,
    df_data: Output[Dataset],
    file_format: str = "csv",
) -> None:
    
    """
    Load data from a specified GCS bucket and file, and return a pandas DataFrame.
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        file_path (str): The path to the file within the GCS bucket.
        file_format (str): The format of the file (e.g., "csv", "json"). Default is "csv".
    
    Returns:
        pd.DataFrame: DataFrame containing the data from the file.
    """
    
    from google.cloud import storage
    import pandas as pd
    from io import BytesIO

    try:
        # Initialize the GCS client.
        storage_client = storage.Client()

        # Get the bucket and blob.
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        file_bytes = blob.download_as_bytes()

        # Load the file into a pandas DataFrame based on the specified format.
        if file_format == "csv":
            df_loaded = pd.read_csv(BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please use 'csv'.")

        if not df_loaded.empty:
            df_loaded.to_csv(df_data.path, index=False)
        else:
            raise ValueError("Table content is empty.")
    except Exception as e:
        print(f"An error occurred during file load: {e}")
        raise


# In[41]:


@dsl.component(packages_to_install=['pandas', 'scikit-learn'])
def preprocess_data(
    df_data: Input[Dataset],
    train_data: Output[Dataset],
) -> None:
    """
    Preprocess the input DataFrame by handling null values in the specified columns.

    Args:
        df_data (pd.DataFrame): Input DataFrame with potential null values.

    Returns:
        pd.DataFrame: DataFrame preprocessed.
    """
    
    from sklearn.impute import SimpleImputer
    import pandas as pd

    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df_imputed = pd.read_csv(df_data.path)
        
        # Preprocess the data
        columns_to_impute = ['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']
        imputer = SimpleImputer(fill_value=0)
        df_imputed[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

        if not df_imputed.empty:
            df_imputed.to_csv(train_data.path, index=False)
        else:
            raise ValueError("Table content is empty.")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise  


# In[42]:


@dsl.component(packages_to_install=['pandas', 'scikit-learn', 'google-cloud-aiplatform'])
def train_and_save_model(
    project_id: str, 
    region: str, 
    model_display_name: str,
    train_data: Input[Dataset]
) -> None:
    """
    Train a Logistic Regression model and save it to the Vertex AI Model Registry and Google Cloud Storage.

    Args:
        df (pd.DataFrame): Input DataFrame for training.
        project_id (str): Google Cloud project ID.
        region (str): Region for Vertex AI.
        model_display_name (str): Display name for the model in Vertex AI.

    Returns:
        None
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from google.cloud import aiplatform
    import pandas as pd
    import joblib
    import os

    try:
        
        df = pd.read_csv(df_data.path)
        
        # Splitting the data into features and target
        X = df.drop('Loan_Approval', axis=1)
        y = df['Loan_Approval']

        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

        # Training the Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        print(y_pred)

        # Save the model to a local file
        model_filename = 'model.joblib'
        joblib.dump(model, model_filename)

        # Initialize the Vertex AI client
        aiplatform.init(project=project_id, location=region)

        # Upload the model to Vertex AI Model Registry
        aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=f'gs://{project_id}/{model_filename}',  # GCS bucket URI for Vertex AI Model Registry
            serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest'
        )

        print(f"Model {model_display_name} successfully uploaded to Vertex AI Model Registry.")

        # Upload the model file to Google Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(project_id)
        blob = bucket.blob(model_filename)
        blob.upload_from_filename(model_filename)

        print(f"Model {model_filename} uploaded to Google Cloud Storage.")

    except Exception as e:
        print(f"An error occurred during model training, upload to Vertex AI, or upload to Cloud Storage: {e}")
        raise  # Re-raise the exception to stop the pipeline


# ## Creating pipeline

# In[43]:


@dsl.pipeline(
    name=PIPELINE_DISPLAY_NAME, 
    description=PIPELINE_DESCRIPTION
)

def main_pipeline(
    data_source: str,
    source_project: str,
    source_dataset: str,
    source_table: str,
    source_bucket: str,
    datafile_name: str,
    train_project_id: str,
    model_display_name: str,
    gcp_region: str = "us-central1",
):
    
    with dsl.Condition(data_source == 'bigquery'):
        load_data_op = load_from_bq(
                                    project_id=source_project,
                                    dataset_id=source_dataset,
                                    table_id=source_table,
                                    location=gcp_region,
                                    ).set_display_name("Load data from BQ")
        preprocess_data_op = preprocess_data(
                                        df_data=load_data_op.outputs['df_data']
                                        ).after(load_data_op).set_display_name("Preprocessing data")
    
        train_save_op = train_and_save_model(
                                        project_id=train_project_id, 
                                        region=gcp_region, 
                                        model_display_name=model_display_name,
                                        train_data=preprocess_data_op.outputs['train_data'], 
                                        ).after(preprocess_data_op).set_display_name("Training and saving model")
    with dsl.Condition(data_source == 'storage'):
        load_data_op = load_from_gcs(
                                    bucket_name=source_bucket,
                                    file_path=datafile_name,
                                    file_format='csv'
                                    ).set_display_name("Load data from GCS")
        
        preprocess_data_op = preprocess_data(
                                        df_data=load_data_op.outputs['df_data']
                                        ).after(load_data_op).set_display_name("Preprocessing data")
    
        train_save_op = train_and_save_model(
                                        project_id=train_project_id, 
                                        region=gcp_region, 
                                        model_display_name=model_display_name,
                                        train_data=preprocess_data_op.outputs['train_data'], 
                                        ).after(preprocess_data_op).set_display_name("Training and saving model")
    
    


# ## Compiling

# In[45]:


compiler.Compiler().compile(
    pipeline_func=main_pipeline,
    package_path='../_execution_/compiled_pipeline.json'
)


# In[ ]:




