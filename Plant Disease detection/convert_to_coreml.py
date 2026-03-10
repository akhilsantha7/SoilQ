import torch
from torchvision import models
import torch.nn as nn
import coremltools as ct

MODEL_PATH = "plant_disease_model_1.pt"
COREML_PATH = "plant_disease_model_1.mlpackage"  # or .mlmodel
NUM_CLASSES = 6

# 1. Load your PyTorch model (ResNet50, same as train.py)
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 2. Trace the model (TorchScript) – input shape must match training (224x224)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 3. Convert traced model to Core ML (mlprogram format by default)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS15
    # You could also specify compute_precision, compute_units, etc.
)

mlmodel.save(COREML_PATH)
print(f"✅ CoreML model saved as {COREML_PATH}")
