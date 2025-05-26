import os
import sys
import glob
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EmotionRecognitionModel
import numpy as np

# Define emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path=None):
    if model_path is None:
        raise ValueError("You must specify the path to the model checkpoint (best_model.pth).")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EmotionRecognitionModel(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    args = parser.parse_args()
    model, device = load_model(args.model_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            input_tensor = preprocess_face(face).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                emotion_idx = np.argmax(probs)
                emotion_label = EMOTIONS[emotion_idx]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{emotion_label}: {probs[emotion_idx]:.2f}', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Facial Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
