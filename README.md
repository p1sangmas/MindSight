# Facial Emotion Recognition (FER) Project

A deep learning application for real-time facial emotion recognition using EfficientNet and Transformer architecture.

## Overview

This project implements a facial emotion recognition system that can detect seven basic emotions: 
angry, disgust, fear, happy, neutral, sad, and surprise. It includes:

- Model training with various techniques
- Real-time emotion detection using webcam
- Mental health assessment dashboard
- Containerized deployment with Docker

## Quick Start

### Option 1: Running with Docker

```bash
# Start the Docker container
./start.sh

# Access the application at:
# http://localhost:8501
```

### Option 2: Running Locally (recommended for webcam functionality)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run src/dashboard_app.py
```

## Camera Access Troubleshooting

If you're having issues with camera access, especially in Docker:

1. **Browser Permissions**:
   - Check that your browser has permission to access your camera
   - Try using Chrome or Firefox (they have better webcam support)
   - Look for the camera icon in your browser's address bar

2. **Docker-specific issues**:
   - Browser-based camera access requires explicit permission
   - If the permission dialog doesn't appear, try clicking the "Request Camera Permission" button in the app
   - See `webcam-setup.md` for detailed Docker camera configuration

3. **Recommended Solution**:
   - For best camera functionality, run the app directly (without Docker)
   - `streamlit run src/dashboard_app.py`

## Model Training

To train your own model:

```bash
python src/train.py --data_dir data --model_folder my_custom_model
```

## Evaluation

```bash
python src/evaluate.py --model_path checkpoints/model_name/best_model.pth
```

## Project Structure

- `src/`: Source code
  - `model.py`: Model architecture
  - `train.py`: Training script
  - `evaluate.py`: Evaluation script
  - `dashboard_app.py`: Streamlit dashboard
- `data/`: Training and testing data
- `checkpoints/`: Saved model weights
- `runs/`: TensorBoard logs
- `docker-compose.yml`: Docker configuration

## License

[MIT License](LICENSE)
