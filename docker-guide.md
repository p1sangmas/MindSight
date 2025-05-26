# Facial Emotion Recognition (FER) - Docker Guide

This document explains how to run the FER application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Quick Start

Run the application with a single command:

```bash
./start.sh
```

Then open your browser and go to: http://localhost:8501

## Manual Setup

### Build and Run

```bash
# Build and start the containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Using Different Model Checkpoints

You can specify a different model checkpoint by setting an environment variable:

1. Edit the `docker-compose.yml` file
2. Add the model path under the environment section:
   ```yaml
   environment:
     - PYTHONUNBUFFERED=1
     - MODEL_PATH=checkpoints/model_kfoldcrossvalidation/best_model.pth
   ```
3. Restart the application: `docker-compose down && docker-compose up -d`

## Camera Access

To use the webcam for emotion recognition:
- Make sure your browser has permission to access your camera
- When running in Docker, the webcam from your host machine must be accessible to the container

## Troubleshooting

### Camera Not Working
Camera access in Docker containers can sometimes be challenging. If you're having issues:
- Try running the application outside Docker for webcam functionality
- For Docker on Linux, you may need to pass additional device flags

### Performance Issues
If the application is running slowly:
- Ensure Docker has adequate resources allocated (check Docker Desktop settings)
- Consider using a machine with GPU support for better performance

## Persisting Data

The model checkpoints are stored in a Docker volume named `model-data`. This ensures your trained models persist between container restarts.
