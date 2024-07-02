# Production ML model training

![Docs](images/pipeline_done.png)

This repository aims to facilitate launching a production training process using Google Cloud Platform (GCP) services. It supports input data from BigQuery or Cloud Storage, and enables deployment of machine learning model to VertexAI endpoint or Cloud Run for FastAPI RESTful API.

## 1. Getting started

This process required the tools showed below:
- [Python]([https://github.com/pyenv/pyenv](https://www.python.org/)) 
- [Docker]([https://github.com/pyenv/pyenv](https://www.docker.com/))
- Kubeflow
- GCP

Additionally, it is required that you create a service account in your GCP project and enable the follow GCP API services:
    - aiplatform.googleapis.com
    - artifactregistry.googleapis.com
    - ml.googleapis.com    
    - bigquery.googleapis.com
    - compute.googleapis.com
    - iamcredentials.googleapis.com
    - iam.googleapis.com
    
## 2. Clone

Clone this repository in your instance:
```bash
git clone https://github.com/omarftt/q-training-gcp-deploy.git
```

## 3. Installation

Create a virtual environment
```bash
virtualenv venv
source venv/bin/activate 
```

Install all packages required
```bash
pip install pip==21.3.1
pip install -r requirements.txt
```

To enable role permission necessary to your Service Account:
```bash
bash installation/users.sh
```

The training process can read two data sources: 'bigquery' or 'storage'. To enable these options, you must upload your .csv file into a cloud storage bucket or a bigquery table. Additionally, if you would like to do it automatically, upload your .csv file into 'data' folder and execute the follow command below:

For bigquery table:
```bash
bash installation/resources.sh bigquery data/dataset.csv {insert_project_name} default-bucket {insert_dataset_name} {insert_table_name}
```
For cloud storage bucket:
```bash
bash installation/resources.sh storage data/dataset.csv {insert_project_name} {insert_bucket_name} default_dataset default_table
```


## 4. Instructions

### 4.1 General Instructions
- All development code modifications must be done into the Kubeflow components in training notebook from workspace folder.

- Modify config.json according to your own resources. The pipeline training could read two data input: 'bigquery' or 'storage' change it according to your preference. 

- Finally, execute the follow command to compile your pipeline:

```bash
jupyter nbconvert workspace/training.ipynb --to python --output ../_execution_/training_compile.py

python _execution_/training_compile.py
```

- Verify that compiled_pipeline.json file has been created on _execution_ folder.

## 5. Execution instructions
Execution will launch a pipeline job in VertexAI Pipelines service. To do that run the follow commands below:

```bash
python _execution_/execute.py
```

## 6. Serving model
To serve the ML model, there are two options available:

### 6.1 Serving model using GCP Vertex AI Endpoint
After check that your trained model is in Model Registry, you can serve the model using Vertex AI service. Run the command below:
```bash
python deploy/endpoint/deploy.py
```

### 6.2 Serving model using Cloud Run
To serve your model with Cloud Run service, there is a FastAPI developed. Please upload your ML model to deploy/api-rest/resources path, enable the Artifact Registry service, create a repository, and update deploy/api-rest/deploy.sh with your resource details. Once done, run the deployment script with sh deploy/api-rest/deploy.sh.
```bash
bash deploy/api-rest/deploy.sh
```

## 7. Arquitecture
To understand better the services used. You can find the architecture here: [Architecture](images/arch.png)
