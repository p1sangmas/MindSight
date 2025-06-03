import os
import sys
import glob
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EmotionRecognitionModel
from src.model_2 import ImprovedEmotionRecognitionModel
import numpy as np

# Define emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path=None, model_version='original'):
    """
    Load the facial emotion recognition model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        model_version: 'original' or 'improved' model architecture
        
    Returns:
        model: The loaded PyTorch model
        device: The device the model is loaded on
    """
    if model_path is None:
        raise ValueError("You must specify the path to the model checkpoint (best_model.pth).")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Select model architecture based on version parameter
    if model_version == 'improved':
        print("Using improved emotion recognition model")
        model = ImprovedEmotionRecognitionModel(num_classes=7).to(device)
    else:
        print("Using original emotion recognition model")
        model = EmotionRecognitionModel(num_classes=7).to(device)
    
    # Load checkpoint - handle both simple state_dict or full checkpoint format
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a full checkpoint with training state
        print(f"Loading from full checkpoint (epoch {checkpoint['epoch']})")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Legacy format - direct state_dict
        print("Loading from legacy model format (state_dict only)")
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, device

def preprocess_face(face):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(face).unsqueeze(0)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--model_version', type=str, default='original', choices=['original', 'improved'],
                        help='Select model architecture (original or improved)')
    parser.add_argument('--display_probs', action='store_true', 
                        help='Display probability values for each emotion')
    parser.add_argument('--top_k', type=int, default=1, 
                        help='Show top K emotion predictions')
    args = parser.parse_args()
    model, device = load_model(args.model_path, args.model_version)
    
    # Print model info
    print("-" * 50)
    print(f"Running Facial Emotion Recognition")
    print(f"Model version: {args.model_version}")
    print(f"Device: {device}")
    print(f"Displaying top {args.top_k} emotions")
    print("Press 'q' to quit")
    print("-" * 50)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Only process the largest face (most prominent one)
        if len(faces) > 0:
            # Find the largest face by area (width * height)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            face = frame[y:y+h, x:x+w]
            input_tensor = preprocess_face(face).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                
                # Get top-k predictions
                top_k = min(args.top_k, len(EMOTIONS))
                top_indices = probs.argsort()[-top_k:][::-1]
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display the primary emotion (highest probability)
                primary_emotion = EMOTIONS[top_indices[0]]
                primary_prob = probs[top_indices[0]]
                
                if args.display_probs:
                    label_text = f'{primary_emotion}: {primary_prob:.2f}'
                else:
                    label_text = primary_emotion
                
                # Add the primary emotion text above the face
                cv2.putText(frame, label_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Display additional emotions if top_k > 1
                if args.top_k > 1:
                    y_offset = 20
                    for i in range(1, len(top_indices)):
                        idx = top_indices[i]
                        if args.display_probs:
                            secondary_text = f'{EMOTIONS[idx]}: {probs[idx]:.2f}'
                        else:
                            secondary_text = EMOTIONS[idx]
                        cv2.putText(frame, secondary_text, (x, y + h + y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        y_offset += 20

        cv2.imshow('Facial Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
