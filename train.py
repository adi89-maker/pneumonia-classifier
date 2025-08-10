import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("Starting training...")

# Dataset paths
base_dir = 'chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Check if directories exist
for directory in [base_dir, train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        print(f"ERROR: Directory '{directory}' not found!")
        print("Please ensure your dataset is organized correctly.")
        exit(1)

print("Dataset directories found ✓")

# Image size and batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
print("Creating data generators...")
try:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
except Exception as e:
    print(f"ERROR creating data generators: {e}")
    print("Please check your dataset structure:")
    print("chest_xray/")
    print("  ├── train/")
    print("  │   ├── NORMAL/")
    print("  │   └── PNEUMONIA/")
    print("  ├── val/")
    print("  │   ├── NORMAL/")
    print("  │   └── PNEUMONIA/")
    print("  └── test/")
    print("      ├── NORMAL/")
    print("      └── PNEUMONIA/")
    exit(1)

# Build model
print("Building model...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled ✓")

# Callbacks
callbacks = [
    ModelCheckpoint(
        'pneumonia_classifier_model.h5',  # This matches what Flask app expects
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
]

# Train the model
print("Starting training...")
try:
    history = model.fit(
        train_generator,
        epochs=10,  # Reduced for faster training
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed successfully!")
    
    # Save the final model (backup)
    model.save('pneumonia_classifier_final.h5')
    print("Models saved:")
    print("- pneumonia_classifier_model.h5 (best model)")
    print("- pneumonia_classifier_final.h5 (final model)")
    
except Exception as e:
    print(f"ERROR during training: {e}")
    exit(1)

# Test the saved model
print("\nTesting saved model...")
try:
    test_model = tf.keras.models.load_model('pneumonia_classifier_model.h5')
    print("✓ Model saved and loaded successfully!")
    print(f"Model input shape: {test_model.input_shape}")
    print(f"Model output shape: {test_model.output_shape}")
except Exception as e:
    print(f"ERROR loading saved model: {e}")

print("\nTraining script completed!")