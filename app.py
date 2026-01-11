import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import joblib
from flask import Flask, request, jsonify
from functools import wraps
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
API_KEY_NAME = "x-api-key"
# We fetch the API KEY from an environment variable for security
# Default is 'default-secret-key' if not set in Docker
API_KEY = os.getenv("API_KEY", "default-secret-key")

app = Flask(__name__)

# --- SECURITY DECORATOR ---
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get(API_KEY_NAME)
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({"detail": "Could not validate credentials"}), 403
    return decorated

# --- MODEL LOADING ---
# Use CPU by default for Docker to avoid complex NVIDIA setup unless specifically configured
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server running on: {device}")

# 1. Load Random Forest Classifier
try:
    rf_classifier = joblib.load("crime_rf_model.pkl")
    print("Random Forest loaded.")
except Exception as e:
    print(f"Error loading RF model: {e}")
    rf_classifier = None

# 2. Load R3D Feature Extractor
try:
    r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
    r3d.fc = nn.Identity() # Remove classification head to get features
    r3d.load_state_dict(torch.load("model_weights.pth", map_location=device), strict=False)
    r3d = r3d.to(device)
    r3d.eval()
    print("R3D model loaded.")
except Exception as e:
    print(f"Error loading R3D model: {e}")
    r3d = None

# --- PREPROCESSING UTILS ---
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

class CrimeVideoDataset(Dataset):
    def __init__(self, video_path, clip_len=16, num_clips=5, size=112):
        self.video_path = video_path
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.size = size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            # Return zeros if video fails to open
            return torch.zeros(self.num_clips, 3, self.clip_len, self.size, self.size)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
        cap.release()

        # Handle short videos (Padding)
        if len(frames) < self.clip_len:
            padding = [torch.zeros(3, self.size, self.size)] * (self.clip_len - len(frames))
            frames.extend(padding)

        # Uniform Sampling
        total_frames = len(frames)
        max_start = max(total_frames - self.clip_len, 0)
        starts = np.linspace(0, max_start, self.num_clips, dtype=int)

        clips = []
        for s in starts:
            if s + self.clip_len <= total_frames:
                clip = torch.stack(frames[s : s+self.clip_len]).permute(1, 0, 2, 3)
                clips.append(clip)
            else:
                clips.append(torch.zeros(3, self.clip_len, self.size, self.size))

        if not clips:
             return torch.zeros(self.num_clips, 3, self.clip_len, self.size, self.size)

        return torch.stack(clips)

def process_video(video_path):
    """Handles dataset creation and feature extraction for a single video file."""
    dataset = CrimeVideoDataset(video_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for clips in loader:
            # Shape: (1, 5, 3, 16, 112, 112)
            # Flatten to pass through R3D
            clips = clips.view(-1, 3, 16, 112, 112).to(device)
            
            # Forward Pass
            feat = r3d.stem(clips)
            for layer in [r3d.layer1, r3d.layer2, r3d.layer3, r3d.layer4]:
                feat = layer(feat)
            
            # Global Average Pooling
            emb = feat.mean([-3, -2, -1])
            
            # Flatten to (1, 2560) for Random Forest
            # 2560 comes from 512 features * 5 clips
            return emb.view(1, -1).cpu().numpy()

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "online",
        "message": "Crime Detection API is running. Use POST /predict to classify videos."
    })

@app.route("/predict", methods=["POST"])
@require_api_key
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # 1. Save Uploaded File Temporarily
    temp_filename = f"temp_{file.filename}"
    try:
        file.save(temp_filename)
        
        # 2. Check if models are loaded
        if r3d is None or rf_classifier is None:
             return jsonify({"error": "Models failed to load on server startup"}), 500

        # 3. Process Video
        features = process_video(temp_filename)
        
        # 4. Predict
        prediction_index = rf_classifier.predict(features)[0]
        probabilities = rf_classifier.predict_proba(features)[0]
        
        result = "Crime" if prediction_index == 1 else "Normal"
        confidence = float(probabilities[prediction_index])
        
        return jsonify({
            "filename": file.filename,
            "prediction": result,
            "confidence": f"{confidence:.2%}",
            "is_crime": bool(prediction_index == 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 5. Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    # For local debugging only
    app.run(host="0.0.0.0", port=8000)
