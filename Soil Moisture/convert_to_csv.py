import pandas as pd

# Replace with your actual file path
txt_file = "Soil.txt"
csv_file = "Soil.csv"

# Read the txt file (assuming it's comma-separated)
df = pd.read_csv(txt_file)

# Save it as CSV
df.to_csv(csv_file, index=False)

print(f"CSV file saved at: {csv_file}")
