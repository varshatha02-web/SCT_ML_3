import os
import shutil
import random

# --- SETTINGS ---
# The folder that currently contains your 'Cat' and 'Dog' folders
INPUT_BASE_DIR = r"D:\skillcraft3\PetImages" 

# The new folder structure will be created here
OUTPUT_BASE_DIR = r"D:\skillcraft3\PetImages_Split"

# The ratio of images to go into the training set (80%)
TRAIN_RATIO = 0.8 
# ----------------

# Create the necessary target directories if they don't exist
def create_dirs():
    for set_name in ['train', 'test']:
        for class_name in ['Cat', 'Dog']:
            os.makedirs(os.path.join(OUTPUT_BASE_DIR, set_name, class_name), exist_ok=True)
    print("Target directories created.")

# Move files into the new structure
def split_data():
    create_dirs()
    
    for class_name in ['Cat', 'Dog']:
        input_class_dir = os.path.join(INPUT_BASE_DIR, class_name)
        
        # 1. Get all file names
        all_files = os.listdir(input_class_dir)
        # Filter out any non-image files if necessary (likeThumbs.db)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
        
        # 2. Shuffle the file names to randomize the split
        random.shuffle(image_files)
        
        # 3. Calculate the split point
        split_point = int(len(image_files) * TRAIN_RATIO)
        
        # 4. Separate into train and test lists
        train_files = image_files[:split_point]
        test_files = image_files[split_point:]

        print(f"\nProcessing {class_name}s: {len(image_files)} total.")
        print(f"  - Training: {len(train_files)} ({TRAIN_RATIO*100:.0f}%)")
        print(f"  - Testing: {len(test_files)} ({(1-TRAIN_RATIO)*100:.0f}%)")

        # 5. Move the files to the new 'train' folder
        for filename in train_files:
            src = os.path.join(input_class_dir, filename)
            dst = os.path.join(OUTPUT_BASE_DIR, 'train', class_name, filename)
            # Use shutil.move to move the file (saves disk space!)
            try:
                shutil.move(src, dst)
            except Exception as e:
                # Catch errors for files that might be corrupted or in use
                print(f"Skipping file move error for {filename}: {e}")
                
        # 6. Move the files to the new 'test' folder
        for filename in test_files:
            src = os.path.join(input_class_dir, filename)
            dst = os.path.join(OUTPUT_BASE_DIR, 'test', class_name, filename)
            try:
                shutil.move(src, dst)
            except Exception as e:
                 print(f"Skipping file move error for {filename}: {e}")

    print("\n✅ Data splitting and file movement complete!")

if __name__ == "__main__":
    split_data()
