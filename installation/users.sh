#!/bin/bash

# Define variables for project ID and service account email
PROJECT_ID="beaming-signal-428023-h8"
SERVICE_ACCOUNT_NAME="sa-experimental-training"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Define an array of roles to be assigned
ROLES=(
    "roles/aiplatform.user"
    "roles/storage.objectViewer"
    "roles/storage.objectCreator"
    "roles/iam.serviceAccountAdmin"
)

# Loop through the roles and add IAM policy bindings
for ROLE in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="$ROLE"
done
