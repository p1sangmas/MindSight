#!/bin/bash

echo "Starting FER (Facial Emotion Recognition) application in Docker..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down

# Build and start the Docker container
echo "Building and starting containers..."
docker-compose up --build -d

echo ""
echo "The application is now running at: http://localhost:8501"
echo ""
echo "=== WEBCAM ACCESS INFORMATION ==="
echo "This version includes a special JavaScript-based webcam solution for Docker."
echo "Make sure to:"
echo "1. Grant camera permissions to your browser when prompted"
echo "2. Use Chrome or Firefox for best compatibility"
echo "3. If camera access fails, try running the app directly with:"
echo "   streamlit run src/dashboard_app.py"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
