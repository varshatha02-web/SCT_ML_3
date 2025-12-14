import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# -----------------------------
# 1. SETTINGS
# -----------------------------
# This path is pointing to your successfully split data
BASE_DIR = r"D:\skillcraft3\PetImages_Split" 

# Directories derived from the base
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMG_SIZE = 128        # Size of images fed to the CNN
BATCH_SIZE = 64      
EPOCHS = 12          

# -----------------------------
# 2. DATA PREPARATION
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,             
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,       
    validation_split=0.2        
)

validation_datagen = ImageDataGenerator(rescale=1./255) # Use a separate generator for validation
test_datagen = ImageDataGenerator(rescale=1./255) # Use this if you want to evaluate on the 'test' folder later

print("Loading and augmenting training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',        
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# -----------------------------
# 3. MODEL ARCHITECTURE (The CNN)
# -----------------------------
model = Sequential([
    # Convolutional Layers: Learn features
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(), # Prepares feature maps for the dense layers
    
    # Dropout layer: Prevents overfitting
    Dropout(0.5), 
    
    # Dense Layers: Make the final classification decision
    Dense(512, activation='relu'),
    
    # Output layer
    Dense(1, activation='sigmoid')
])

# -----------------------------
# 4. COMPILATION AND TRAINING
# -----------------------------

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nStarting CNN Training (Target: 90%+ Accuracy)... This will take ~30-60 minutes.")

# The training step
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
)

# -----------------------------
# 5. PREDICT IMAGE (Using OpenCV for fast loading)
# -----------------------------

def predict_image(filename):
    if not os.path.exists(filename):
        print(f"❌ '{filename}' not found! Please place 'test.jpg' in the script folder.")
        return

    # Load, resize, and prepare the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert OpenCV BGR to Matplotlib RGB
    
    img_normalized = img / 255.0
    img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

    # Get prediction (outputs a single probability value between 0 and 1)
    prob = model.predict(img_input, verbose=0)[0][0]
    
    # Determine label and probabilities
    dog_prob = prob * 100
    cat_prob = (1 - prob) * 100
    label = "Dog" if prob >= 0.5 else "Cat"

    # SHOW IMAGE + BAR GRAPH
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Image Display
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    # Bar Graph Display
    probabilities = [cat_prob, dog_prob]
    bars = ax[1].bar(["Cat", "Dog"], probabilities, color=["blue", "orange"])
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("Probability (%)")
    ax[1].set_title(f"Prediction: {label} (Confidence: {max(probabilities):.1f}%)")

    # Add percentages above bars
    for bar, p in zip(bars, probabilities):
        ax[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{p:.1f}%", ha='center', va='bottom', fontsize=12)

    plt.show()
    
    # Get the final validation accuracy from the training history
    final_accuracy = history.history['val_accuracy'][-1] * 100
    
    print("\n--- SINGLE IMAGE PREDICTION (CNN) ---")
    print(f"Prediction: {label}")
    print(f"Cat Probability: {cat_prob:.1f}%")
    print(f"Dog Probability: {dog_prob:.1f}%")
    print(f"--- Final Validation Accuracy: {final_accuracy:.2f}% ---")
    
    return label

# -----------------------------
# 6. RUN PREDICTION
# -----------------------------
test_image = "test.jpg" 
predict_image(test_image)
