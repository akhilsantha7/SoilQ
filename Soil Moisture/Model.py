import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import os


# 1. Load dataset
df = pd.read_csv("Soil.csv")

# -----------------------------
# Model 1: Classification for Irrigation_Needed
# -----------------------------
# X1 = df.drop(columns=["Irrigation_Needed", "Time_to_Irrigation_hr", "Fertilizer_Type"])
X1 = df[["Soil_Moisture", "Soil_Temp", "Soil_pH", "Light"]]
# y1 = df["Irrigation_Needed"]
y1 = df["Irrigation_Needed"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

clf_irrigation = RandomForestClassifier(n_estimators=200, random_state=42)
clf_irrigation.fit(X1_train, y1_train)

y1_pred = clf_irrigation.predict(X1_test)
print("\n--- Irrigation Needed Classification ---")
print("Accuracy:", accuracy_score(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

# -----------------------------
# Model 2: Regression for Time_to_Irrigation_hr
# -----------------------------
# X2 = df.drop(columns=["Time_to_Irrigation_hr", "Fertilizer_Type", "Irrigation_Needed"])  
X2 = df[["Soil_Moisture", "Soil_Temp", "Soil_pH", "Light"]]

# ✅ Removed Irrigation_Needed so input only depends on sensors
y2 = df["Time_to_Irrigation_hr"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

reg_irrigation = RandomForestRegressor(n_estimators=200, random_state=42)
reg_irrigation.fit(X2_train, y2_train)

y2_pred = reg_irrigation.predict(X2_test)
print("\n--- Time to Irrigation Regression ---")
print("MAE:", mean_absolute_error(y2_test, y2_pred))
print("R² Score:", r2_score(y2_test, y2_pred))

# -----------------------------
# Model 3: Classification for Fertilizer_Type
# -----------------------------
# X3 = df.drop(columns=["Fertilizer_Type", "Time_to_Irrigation_hr"])  
# # ✅ Keep Irrigation_Needed here (it helps predict fertilizer)
# y3 = df["Fertilizer_Type"]

# X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

# clf_fertilizer = RandomForestClassifier(n_estimators=200, random_state=42)
# clf_fertilizer.fit(X3_train, y3_train)

# y3_pred = clf_fertilizer.predict(X3_test)
# print("\n--- Fertilizer Type Classification ---")
# print("Accuracy:", accuracy_score(y3_test, y3_pred))
# print(classification_report(y3_test, y3_pred))

out_dir = "models/"
os.makedirs(out_dir, exist_ok=True)


import pickle

# Save all three models
with open(out_dir + "clf_irrigation.pkl", "wb") as f:
    pickle.dump(clf_irrigation, f)

with open(out_dir + "reg_irrigation.pkl", "wb") as f:
    pickle.dump(reg_irrigation, f)

# with open(out_dir + "clf_fertilizer.pkl", "wb") as f:
#     pickle.dump(clf_fertilizer, f)

print("✅ Models saved as .pkl files")

# -----------------------------
# Example Prediction
# -----------------------------


sample_data = {
    "Soil_Moisture": [40],
    "Soil_Temp": [28],
    "Soil_pH": [6.8],
    "Light": [900],
    # "Nitrogen": [150],
    # "Phosphorus": [80],
    # "Potassium": [200],
    # "Irrigation_Needed": [0]  # only used in fertilizer model
}

# # Build DataFrames for each model
# sample_X1 = pd.DataFrame(sample_data).drop(columns=["Irrigation_Needed"])  # model 1
# sample_X2 = pd.DataFrame(sample_data).drop(columns=["Irrigation_Needed"])  # model 2
# sample_X3 = pd.DataFrame(sample_data)    
sample_X = pd.DataFrame(sample_data)
print("Irrigation Needed:", clf_irrigation.predict(sample_X)[0])
print("Time to Irrigation (hr):", reg_irrigation.predict(sample_X)[0])                                  # model 3

# print("\n--- Example Prediction with Sample Data ---")
# print("Irrigation Needed:", clf_irrigation.predict(sample_X1)[0])
# print("Time to Irrigation (hr):", reg_irrigation.predict(sample_X2)[0])
# print("Fertilizer Type:", clf_fertilizer.predict(sample_X3)[0])

import pdb
pdb.set_trace()
