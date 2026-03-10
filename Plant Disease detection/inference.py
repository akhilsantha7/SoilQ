import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from PIL import Image
import sys

# --- CONFIG ---
MODEL_PATH = "plant_disease_model_1.pt"
IMAGE_PATH = "11.jpg"  # Change to your test image path

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRANSFORM (must match training: 256 resize, 224 center crop, ImageNet norm) ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- LOAD MODEL (ResNet50, same architecture as train.py) ---
NUM_CLASSES = 6
weights = ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- LOAD IMAGE ---
img = Image.open(IMAGE_PATH).convert("RGB")
img_t = transform(img).unsqueeze(0).to(device)

class_names = [
    "Healthy",
    "Anthracnose",
    "Powdered Mildew",
    "Sun Blotch",
    "Cercospora Spot",
    "Root Rot",
]

# --- INFERENCE ---
with torch.no_grad():
    outputs = model(img_t)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    _, predicted = torch.max(outputs, 1)
    pred_class = predicted.item()

print(f"Predicted class: {pred_class} ({class_names[pred_class]})")
print("Class probabilities:")
for idx, prob in enumerate(probabilities):
    print(f"  {class_names[idx]}: {prob:.4f}")
