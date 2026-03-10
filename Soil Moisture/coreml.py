import pickle
import coremltools as ct
import numpy as np

# -----------------------------
# Irrigation Needed Classifier
# -----------------------------
with open("models/clf_irrigation.pkl", "rb") as f:
    clf_irrigation = pickle.load(f)

model_irrigation = ct.converters.sklearn.convert(
    clf_irrigation,
    input_features=["Soil_Moisture", "Soil_Temp", "Soil_pH", "Light"]
)
model_irrigation.save("IrrigationNeeded.mlmodel")

# -----------------------------
# Fertilizer Type Classifier
# -----------------------------
# with open("models/clf_fertilizer.pkl", "rb") as f:
#     clf_fertilizer = pickle.load(f)

# model_fertilizer = ct.converters.sklearn.convert(
#     clf_fertilizer,
#     input_features=["Soil_Moisture", "Soil_Temp", "Soil_pH", "Light", "Nitrogen", "Phosphorus", "Potassium", "Irrigation_Needed"]
# )
# model_fertilizer.save("FertilizerType.mlmodel")

# -----------------------------
# Time to Irrigation Regressor 
# -----------------------------
with open("models/reg_irrigation.pkl", "rb") as f:
    reg_irrigation = pickle.load(f)

model_time = ct.converters.sklearn.convert(
    reg_irrigation,
    input_features=[
        "Soil_Moisture", "Soil_Temp", "Soil_pH",
        "Light"
    ],
    output_feature_names="Time_to_Irrigation_hr"
)
model_time.save("TimeToIrrigation.mlmodel")

print("✅ All 3 models converted to Core ML successfully!")