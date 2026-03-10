import torch
from torchvision import models
import coremltools as ct
from PIL import Image
import numpy as np
import os

# --- CONFIG ---
PYTORCH_MODEL_PATH = "plant_disease_model_1.pt"
COREML_MODEL_PATH = "plant_disease_model_1_ios15.mlpackage"
TEST_IMAGE_PATH = "test_rr.jpg"

# --- CLASS LABELS ---
class_names = [
    "Healthy",
    "Anthracnose",
    "Powdered Mildew",
    "Sun Blotch",
    "Cercospora Spot",
    "Root Rot"
]

# --- 1. LOAD PYTORCH MODEL ---
print("🔹 Loading PyTorch model...")
model = models.vgg16(pretrained=False)
n_features = model.classifier[0].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(n_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, len(class_names))
)

try:
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location="cpu"))
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- 2. TRACE MODEL ---
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# --- 3. CONVERT TO CORE ML (iOS 15+, with class probabilities) ---
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        name="input",
        shape=(1, 3, 224, 224),
        scale=1/255.0,
        bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]
    )],
    classifier_config=ct.ClassifierConfig(class_labels=class_names),
    minimum_deployment_target=ct.target.iOS15
)

# Optional metadata
mlmodel.short_description = "Plant disease classifier"
mlmodel.input_description["input"] = "Image of a plant leaf"
mlmodel.output_description["classLabel"] = "Predicted disease class"

mlmodel.save(COREML_MODEL_PATH)
print(f"✅ Core ML iOS15+ model saved at {COREML_MODEL_PATH}")

# --- 4. OPTIONAL: TEST INFERENCE ---
def run_test(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    prediction = mlmodel.predict({"input": image})

    pred_class = prediction.get('classLabel', 'Unknown')
    print(f"\n✅ Predicted class: {pred_class}")

    class_probs = prediction.get('classLabel_probs')
    if class_probs:
        # Convert logits to probabilities
        labels, logits = zip(*class_probs.items())
        logits = np.array(logits, dtype=np.float32)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        print("🔍 Class probabilities (softmax):")
        for label, prob in sorted(zip(labels, probs), key=lambda x: x[1], reverse=True):
            print(f"{label}: {prob:.4f}")
    else:
        print("⚠️ classLabel_probs not found.")

# Example usage
run_test(TEST_IMAGE_PATH)
