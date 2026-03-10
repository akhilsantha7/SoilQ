import os
from PIL import Image

DATA_DIR = "sb"
OUTPUT_DIR = "sb_new"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through all folders in dataset
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    
    if not os.path.isdir(folder_path):
        continue
    
    # Create corresponding folder inside OUTPUT_DIR
    output_folder = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing folder: {folder}")
    count = 1
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # Skip non-image files
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        
        try:
            # Open and convert to RGB
            img = Image.open(file_path).convert("RGB")
            
            # New filename
            new_filename = f"{count}.jpg"
            new_path = os.path.join(output_folder, new_filename)
            
            # Save as JPG
            img.save(new_path, "JPEG", quality=95)
            count += 1
        
        except Exception as e:
            print(f"❌ Error with {file}: {e}")

print("✅ All images converted and saved in 'data_jpg/'")
