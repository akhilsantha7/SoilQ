import coremltools as ct
import numpy as np
from PIL import Image

# --- CONFIG ---
MODEL_PATH = "plant_disease_model_1.mlpackage"  # or .mlmodel
IMAGE_PATH = "test_rr.jpg"

# --- CLASS LABELS ---
class_names = [
    "Healthy",
    "Anthracnose",
    "Powdered Mildew",
    "Sun Blotch",
    "Cercospora Spot",
    "Root Rot"
]

# --- LOAD MODEL ---
print("🔹 Loading CoreML model...")
mlmodel = ct.models.MLModel(MODEL_PATH)

# --- INSPECT MODEL I/O ---
spec = mlmodel.get_spec()
input_names = [i.name for i in spec.description.input]
output_names = [o.name for o in spec.description.output]
print(f"📥 Input names: {input_names}")
print(f"📤 Output names: {output_names}")

# --- IMAGE PREPROCESSING (same as PyTorch training) ---
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std

    # HWC → CHW + batch dimension
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

# --- LOAD IMAGE ---
input_data = preprocess_image(IMAGE_PATH)

# --- RUN INFERENCE ---
input_name = input_names[0]
print(f"🚀 Running inference on {IMAGE_PATH}...")
prediction = mlmodel.predict({input_name: input_data})

print("🔎 Prediction keys:", prediction.keys())

# --- FIND OUTPUT KEY AUTOMATICALLY ---
output_key = None
for k, v in prediction.items():
    if isinstance(v, (list, np.ndarray)):
        output_key = k
        break

if output_key is None:
    raise ValueError("❌ No numeric output found in prediction dictionary.")

# --- PROCESS OUTPUT ---
output = np.array(prediction[output_key])[0]

# Apply softmax (if not already probabilities)
if not np.allclose(np.sum(output), 1.0, atol=1e-3):
    probabilities = np.exp(output) / np.sum(np.exp(output))
else:
    probabilities = output

pred_class = np.argmax(probabilities)

# --- PRINT RESULTS ---
print(f"\n✅ Predicted class: {pred_class} ({class_names[pred_class]})\n")
print("Class probabilities:")
for i, p in enumerate(probabilities):
    print(f"{class_names[i]}: {p:.4f}")
