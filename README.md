🐱🐶 SkillCraft Task 3 – Image Classification (Cats vs Dogs)

This project implements an image classification model to distinguish between cats and dogs using the PetImages dataset.
While the task originally involved a traditional ML approach (SVM), a CNN-based deep learning model was used to handle the complexity of image data and achieved 90%+ accuracy.

📁 Files Included

PetImages/ – Original dataset

PetImages_Split/ – Train–test split dataset

split_data.py – Dataset splitting script

svm_cats_dogs.py – CNN model implementation

🧩 Model Approach

Images are resized and normalized before training.
A Convolutional Neural Network automatically learns visual features for accurate classification.

🚀 Workflow Overview

Split dataset into training and testing sets.

Train a CNN model using TensorFlow/Keras.

Predict and visualize results with confidence scores.

🧾 Output

Classification accuracy above 90%

Single-image prediction with probability visualization
