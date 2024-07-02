#!/bin/bash

# Set environment variables for parameters
PLATFORM="linux/amd64"
IMAGE_NAME="beaming-signal-428023-h8/deploy-fastapi/inference"
TAG=latest

# Build the Docker image
docker buildx build --platform $PLATFORM -t $IMAGE_NAME:$TAG .

# Push the Docker image
docker push $IMAGE_NAME:$TAG