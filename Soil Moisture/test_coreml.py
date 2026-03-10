import coremltools as ct
import numpy as np

# Sample input data
sample = {
    "Soil_Moisture": 0.25,
    "Soil_Temp": 22.5,
    "Soil_pH": 6.8,
    "Light": 300,
    "Nitrogen": 15,
    "Phosphorus": 10,
    "Potassium": 20,
    "Irrigation_Needed": 1  # Only for FertilizerType model
}

# Load IrrigationNeeded model and predict
model_irrigation = ct.models.MLModel("IrrigationNeeded.mlmodel")
out_irrigation = model_irrigation.predict({k: sample[k] for k in [
    "Soil_Moisture", "Soil_Temp", "Soil_pH", "Light", "Nitrogen", "Phosphorus", "Potassium"
]})
print("IrrigationNeeded.mlmodel output:", out_irrigation)

# Load FertilizerType model and predict
model_fertilizer = ct.models.MLModel("FertilizerType.mlmodel")
out_fertilizer = model_fertilizer.predict({k: sample[k] for k in [
    "Soil_Moisture", "Soil_Temp", "Soil_pH", "Light", "Nitrogen", "Phosphorus", "Potassium", "Irrigation_Needed"
]})
print("FertilizerType.mlmodel output:", out_fertilizer)

# Load TimeToIrrigation model and predict
model_time = ct.models.MLModel("TimeToIrrigation.mlmodel")
out_time = model_time.predict({k: sample[k] for k in [
    "Soil_Moisture", "Soil_Temp", "Soil_pH", "Light", "Nitrogen", "Phosphorus", "Potassium"
]})
print("TimeToIrrigation.mlmodel output:", out_time)