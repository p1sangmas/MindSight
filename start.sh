#!/bin/bash

echo "Starting FER (Facial Emotion Recognition) application in Docker..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Print camera information for the user
echo "Note: This application uses OpenCV to access your webcam."
echo "For Docker webcam access to work, your camera device must be mapped properly to the container."

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down

# Build and start the Docker container
echo "Building and starting containers..."
docker-compose up --build -d

echo ""
echo "The application is now running at: http://localhost:8501"
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
