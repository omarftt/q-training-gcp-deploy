# Production ML model training

The purpose of this repository is to launch a training process in production using GCP cloud services. To do that inputs could be from bigquery or cloud storage and ML model can be available in VertexAI endpoint or FastAPI Restful API.

## 1. Getting started

This process required the tools showed below:
- [Python]([https://github.com/pyenv/pyenv](https://www.python.org/)) 
- [Docker]([https://github.com/pyenv/pyenv](https://www.docker.com/))
- Kubeflow
- GCP

Additionally, it is required that you create a service account in your GCP project and enable the follow GCP API services:

## 2. Clone

Clone this repository in your instance:
```bash
git clone https://github.com/omarftt/API-FastAPI-Postgres.git
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
```bash
For bigquery table:
bash installation/resources.sh bigquery data/dataset.csv {insert_project_name} default-bucket {insert_dataset_name} {insert_table_name}

For cloud storage bucket:
bash installation/resources.sh storage data/dataset.csv {insert_project_name} {insert_bucket_name} default_dataset default_table
```


## 4. Instructions

### 4.1 General Instructions
- All development code modifications must be done into the Kubeflow components in training notebook from workspace folder.

- Modify config.json according to your own resources. The pipeline training could read two data input: 'bigquery' or 'storage' change it according to your preference. 

- Finally, execute the follow command to compile your pipeline:

```bash
jupyter nbconvert workspace/training.ipynb --to python --output ../_execution_/training_compile.py

python _execution_/compile.py
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

### 6.2 Serving model using FastAPI

## 7. Arquitecture explanation