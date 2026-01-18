# Import TensorFlow for building and training deep learning models
import tensorflow as tf

# Import NumPy for numerical operations
import numpy as np

# Import EarlyStopping callback to stop training when validation performance stops improving
from tensorflow.keras.callbacks import EarlyStopping

# Import Sequential model to stack layers line-by-line
from tensorflow.keras.models import Sequential

# Import CNN-related layers
from tensorflow.keras.layers import (
    Conv2D,        # Performs convolution to extract spatial features
    MaxPooling2D,  # Reduces spatial dimensions of feature maps
    Dense,         # Fully connected neural network layer
    Flatten,       # Converts 2D feature maps into 1D feature vector
    Dropout,       # Randomly disables neurons to reduce overfitting
    Rescaling      # Scales pixel values to a smaller range
)

# Enable debug mode for tf.data pipelines (helps identify dataset-related issues)
tf.data.experimental.enable_debug_mode()

# Data augmentation model to increase dataset diversity
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),   # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.2),        # Randomly rotate images
    tf.keras.layers.RandomZoom(0.2),            # Randomly zoom images
    tf.keras.layers.RandomContrast(0.2),        # Randomly adjust image contrast
])

# Define image dimensions for model input
IMG_SIZE = (128, 128)

# Define number of samples processed per batch
BATCH_SIZE = 32

# Seed value for reproducible dataset splitting
SEED = 42

# Total number of training epochs
EPOCHS = 50

# Load training dataset from directory with automatic labels
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"/kaggle/input/kaggle-cat-vs-dog-dataset/kagglecatsanddogs_3367a/PetImages",  # Dataset path
    validation_split=0.2,        # Use 20% data for validation
    subset="training",           # Specify training subset
    seed=SEED,                   # Seed for consistent split
    image_size=IMG_SIZE,         # Resize images
    batch_size=BATCH_SIZE,       # Batch size
    color_mode="rgb",            # Load images in RGB format
    interpolation="bilinear"     # Resize interpolation method
)

# Load validation dataset from same directory
val_ds = tf.keras.utils.image_dataset_from_directory(
    r"/kaggle/input/kaggle-cat-vs-dog-dataset/kagglecatsanddogs_3367a/PetImages",  # Dataset path
    validation_split=0.2,        # Use same validation split
    subset="validation",         # Specify validation subset
    seed=SEED,                   # Same seed for consistency
    image_size=IMG_SIZE,         # Resize images
    batch_size=BATCH_SIZE,       # Batch size
    color_mode="rgb",            # RGB images
    interpolation="bilinear"     # Resize interpolation
)

# Print detected class labels
print("Classes:", train_ds.class_names)  # ['Cat', 'Dog']

# Ignore corrupted or unreadable images in dataset
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds   = val_ds.apply(tf.data.experimental.ignore_errors())

# Create normalization layer to scale pixel values between 0 and 1
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to training data
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Apply normalization to validation data
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Automatically tune dataset performance
AUTOTUNE = tf.data.AUTOTUNE

# Shuffle, repeat, cache, and prefetch training dataset for performance
train_ds = train_ds.shuffle(1000).repeat().cache().prefetch(AUTOTUNE)

# Cache and prefetch validation dataset
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Build CNN model using Sequential API
cnn = Sequential([
    tf.keras.layers.Input(shape=(128,128,3)),  # Define input shape for images
    data_augmentation,                         # Apply data augmentation only during training

    Conv2D(32, (3,3), activation='relu', padding='same'),  # First convolution layer
    MaxPooling2D(),                              # Reduce spatial size

    Conv2D(64, (3,3), activation='relu', padding='same'),  # Second convolution layer
    MaxPooling2D(),                              # Downsampling

    Conv2D(128, (3,3), activation='relu', padding='same'), # Third convolution layer
    MaxPooling2D(),                              # Downsampling

    Conv2D(256, (3,3), activation='relu', padding='same'), # Fourth convolution layer
    MaxPooling2D(),                              # Downsampling

    Flatten(),                                  # Convert feature maps to 1D vector
    Dense(256, activation='relu'),              # Fully connected layer
    Dropout(0.5),                               # Drop 50% neurons to prevent overfitting
    Dense(1, activation='sigmoid')              # Output layer for binary classification
])

# Compile the CNN model
cnn.compile(
    optimizer='adam',                   # Adaptive optimizer
    loss='binary_crossentropy',         # Loss function for binary classification
    metrics=['accuracy']                # Metric to evaluate model performance
)

# Display CNN architecture summary
cnn.summary()

# Define EarlyStopping to stop training when validation loss stops improving
early_stop = EarlyStopping(
    monitor='val_loss',                 # Monitor validation loss
    patience=3,                         # Stop after 3 epochs with no improvement
    restore_best_weights=True           # Restore best model weights
)

# Train the CNN model
history = cnn.fit(
    train_ds,                           # Training dataset
    validation_data=val_ds,             # Validation dataset
    epochs=EPOCHS,                      # Maximum number of epochs
    callbacks=[early_stop]              # Apply early stopping
)

# Save the trained CNN model to disk
cnn.save('dog-vs-cat-classifier5.h5')

# Confirm successful save
print("File saved")