#!/bin/bash

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <bigquery|storage> <csv_file> [<bucket_name>] [<dataset_name>] [<table_name>]"
    exit 1
fi

# Set variables from arguments
COMMAND=$1
CSV_FILE=$2
PROJECT_ID=$3
BUCKET_NAME=${4:-my-default-bucket}
DATASET_NAME=${5:-my_default_dataset}
TABLE_NAME=${6:-my_default_table}
LOCATION="us-central1"

# Ensure gcloud and bq commands are available
if ! command -v gcloud &> /dev/null || ! command -v bq &> /dev/null; then
    echo "gcloud and bq commands are required but not installed."
    exit 1
fi

# Function to upload CSV to BigQuery
upload_to_bigquery() {
    # Create the dataset if it doesn't exist
    bq --location="$LOCATION" mk -d --description "My dataset" "$PROJECT_ID:$DATASET_NAME"

    # Upload the CSV to BigQuery (create or replace the table)
    bq --location="$LOCATION" load --replace --autodetect --source_format=CSV "$PROJECT_ID:$DATASET_NAME.$TABLE_NAME" "$CSV_FILE"
    
    if [ $? -eq 0 ]; then
        echo "CSV uploaded to BigQuery successfully."
    else
        echo "Failed to upload CSV to BigQuery."
        exit 1
    fi
}

# Function to upload CSV to Cloud Storage
upload_to_storage() {
    # Create the bucket if it doesn't exist
    gsutil mb -p "$PROJECT_ID" -l US "gs://$BUCKET_NAME" 2> /dev/null

    # Upload the CSV file to the bucket
    gsutil cp "$CSV_FILE" "gs://$BUCKET_NAME/"

    if [ $? -eq 0 ]; then
        echo "CSV uploaded to Cloud Storage successfully."
    else
        echo "Failed to upload CSV to Cloud Storage."
        exit 1
    fi
}

# Main command switch
case "$COMMAND" in
    bigquery)
        upload_to_bigquery
        ;;
    storage)
        upload_to_storage
        ;;
    *)
        echo "Invalid command. Use 'bigquery' or 'storage'."
        exit 1
        ;;
esac
