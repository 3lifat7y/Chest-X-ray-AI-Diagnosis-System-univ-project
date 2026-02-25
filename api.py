from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# List of disease classes in the same order used during model training
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Pleural_Thickening', 'No Finding'
]

# Dictionary of short descriptions for each disease
DESCRIPTIONS = {
    'Atelectasis': 'Partial or complete collapse of the lung.',
    'Cardiomegaly': 'Enlargement of the heart.',
    'Consolidation': 'Filling of the lung with liquid or solid material.',
    'Edema': 'Accumulation of fluid in the lung tissue.',
    'Effusion': 'Excess fluid in the pleural cavity.',
    'Emphysema': 'Damage to the alveoli causing breathing difficulty.',
    'Fibrosis': 'Scarring of lung tissue.',
    'Hernia': 'Protrusion of tissue through the diaphragm.',
    'Infiltration': 'Abnormal substance in lung tissue.',
    'Mass': 'Lump or growth in the lung.',
    'Nodule': 'Small growth or mass in the lung.',
    'Pneumonia': 'Infection causing lung inflammation.',
    'Pneumothorax': 'Collapsed lung due to air in the pleural cavity.',
    'Pleural_Thickening': 'Thickening of the pleural lining.',
    'No Finding': 'No detectable abnormalities.'
}

# Path to the trained model file
model_path = 'D:/university/LEVEL 2/Semster 2/project/code/output/best_model_multi_label_nih_original_resnet50_v3.pth'

# Use GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ResNet50 model and modify the final layer for multi-label classification
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_classes = len(CLASSES)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

# Load model weights from file
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(model_state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define image preprocessing steps (must match training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_likely_chest_xray(image):
    """
    Basic check to determine if an image is likely a chest X-ray.
    Assumes chest X-rays are typically grayscale or have low color variance.
    """
    # Convert image to numpy array
    img_array = np.array(image.convert('RGB'))
    # Calculate standard deviation of color channels
    channel_std = np.std(img_array, axis=(0, 1))
    # If standard deviation across RGB channels is low, it's likely grayscale
    is_grayscale = np.all(channel_std < 20)  # Threshold for low color variance
    return is_grayscale

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Check if the image is likely a chest X-ray
        if not is_likely_chest_xray(image):
            return jsonify({"error": "Please upload a chest X-ray image."}), 400

        # Apply preprocessing
        image = transform(image).unsqueeze(0).to(device)

        # Run model inference without computing gradients
        with torch.no_grad():
            outputs = model(image)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # Get diseases with probability above threshold (e.g., 0.5)
        threshold = 0.5
        disease_probs = [(CLASSES[i], probs[i]) for i in range(len(CLASSES)) if probs[i] >= threshold and CLASSES[i] != 'No Finding']
        
        # Sort by probability in descending order and take top 2
        disease_probs = sorted(disease_probs, key=lambda x: x[1], reverse=True)[:2]
        
        # If no diseases are above threshold, check for 'No Finding'
        if not disease_probs:
            max_idx = np.argmax(probs)
            if CLASSES[max_idx] == 'No Finding':
                return jsonify({
                    "diseases": [{"name": "No Finding", "description": DESCRIPTIONS["No Finding"]}],
                    "num_diseases": 0
                })
            else:
                # Take the top disease even if below threshold
                disease_probs = [(CLASSES[max_idx], probs[max_idx])]

        # Prepare response with up to two diseases
        result = {
            "diseases": [
                {"name": disease, "description": DESCRIPTIONS[disease]}
                for disease, _ in disease_probs
            ],
            "num_diseases": len(disease_probs)
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(host='localhost', port=5000, debug=True)