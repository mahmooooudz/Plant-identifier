# train_layer1.py - FULLY FIXED VERSION
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import time

# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from scripts.utils import plot_training_history, create_visual_samples, get_class_weights

# Default settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LEARNING_RATE = 0.00001
DROPOUT_RATE = 0.3
L2_REG = 0.001
UNFREEZE_LAYERS = 80

def create_model(input_shape=(224, 224, 3), learning_rate=0.0001, dropout_rate=0.3, l2_reg=0.001):
    """Create a fine-tuned MobileNetV2 model for binary plant classification"""
    # Load the MobileNetV2 base model with pre-trained weights
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification layers with enhanced regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)  # Add dropout for regularization
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)  # Add dropout for regularization
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model, base_model

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """Create training and validation data generators with enhanced augmentation"""
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,         # Rotate up to 40 degrees
        width_shift_range=0.3,     # Shift horizontally up to 30%
        height_shift_range=0.3,    # Shift vertically up to 30%
        shear_range=0.3,           # Shear angle in counter-clockwise direction
        zoom_range=[0.7, 1.3],     # Zoom in/out by 30%
        horizontal_flip=True,      # Randomly flip horizontally
        vertical_flip=True,        # Randomly flip vertically (useful for plants)
        fill_mode='nearest',       # Fill any newly created pixels
        brightness_range=[0.7, 1.3], # Adjust brightness by 30%
        channel_shift_range=30,    # Shift color channels
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Flow from directory for training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Flow from directory for validation data
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def unfreeze_and_fine_tune(model, base_model, learning_rate=0.00001, unfreeze_layers=80):
    """Unfreeze layers for fine-tuning - FIXED VERSION"""
    # Get the base model more robustly
    if hasattr(model, 'layers') and len(model.layers) > 0:
        # Try to find the base model in the layers
        base_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_layer = layer
                break
        
        if base_layer is not None:
            print(f"Found base model: {base_layer.name}")
            base_layer.trainable = True
            
            # Freeze early layers, unfreeze later layers
            total_layers = len(base_layer.layers)
            freeze_until = max(0, total_layers - unfreeze_layers)
            
            for i, layer in enumerate(base_layer.layers):
                layer.trainable = i >= freeze_until
                
            print(f"Made last {unfreeze_layers} of {total_layers} layers trainable")
        else:
            print("Base model not found as a layer, using alternative approach")
            # Alternative approach: make all layers trainable
            for layer in model.layers:
                layer.trainable = True
    else:
        print("Model has no layers, using direct approach")
        # Direct approach - make the model trainable
        model.trainable = True
    
    # Print trainable status
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'trainable'):
            print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")
    
    # Re-compile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def evaluate_model(model, validation_generator, model_dir):
    """Evaluate the model and generate classification report"""
    # Predict classes for validation data
    validation_generator.reset()
    y_pred = model.predict(validation_generator, verbose=1)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # True classes
    y_true = validation_generator.classes
    
    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=['Not Plant', 'Plant'])
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save classification report to file
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Not Plant', 'Plant'], rotation=45)
    plt.yticks(tick_marks, ['Not Plant', 'Plant'])
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Create visual samples of correct and incorrect predictions
    create_visual_samples(validation_generator, y_pred, model_dir)
    
    return report, cm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a plant detection model')
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='Directory containing the training data')
    parser.add_argument('--model_dir', type=str, default='models/layer1_is_plant',
                        help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--fine_tune_epochs', type=int, default=15,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        help='Image dimensions (width, height)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for training')
    parser.add_argument('--fine_tune_lr', type=float, default=0.00001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--no_fine_tune', action='store_true',
                        help='Skip fine-tuning phase')
    parser.add_argument('--class_weight', action='store_true', default=True,
                        help='Use class weights to handle imbalanced data')
    
    # Add new parameters for data augmentation and regularization
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--l2_reg', type=float, default=0.001,
                        help='L2 regularization strength')
    parser.add_argument('--unfreeze_layers', type=int, default=80, 
                        help='Number of layers to unfreeze during fine-tuning')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, args.data_dir)
    model_dir = os.path.join(parent_dir, args.model_dir)
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Print the training configuration
    print("\n==== Plant Detection Model Training ====")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Initial epochs: {args.epochs}")
    if not args.no_fine_tune:
        print(f"Fine-tuning epochs: {args.fine_tune_epochs}")
    else:
        print("Fine-tuning: Disabled")
    print(f"Learning rate: {args.learning_rate}")
    if not args.no_fine_tune:
        print(f"Fine-tuning learning rate: {args.fine_tune_lr}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"L2 regularization: {args.l2_reg}")
    print(f"Layers to unfreeze: {args.unfreeze_layers}")
    print(f"Class weights: {'Enabled' if args.class_weight else 'Disabled'}")
    print("=======================================\n")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Create data generators
    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators(
        data_dir, 
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )
    
    # Calculate class weights if enabled
    class_weights = None
    if args.class_weight:
        print("Calculating class weights...")
        class_weights = get_class_weights(train_generator.classes)
        print(f"Class weights: {class_weights}")
    
    # Create the model
    print("Creating model...")
    model, base_model = create_model(
        input_shape=(*args.img_size, 3),
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Initial training phase
    print("\nStarting initial training phase...")
    t_start = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        class_weight=class_weights
    )
    t_end = time.time()
    print(f"Initial training completed in {(t_end - t_start)/60:.2f} minutes")
    
    # Plot training history
    plot_training_history(history, model_dir, "initial_training")
    
    # Fine-tuning phase
    if not args.no_fine_tune:
        try:
            print("\nStarting fine-tuning phase...")
            # Load the best weights from initial training
            if os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                model.load_weights(os.path.join(model_dir, 'best_model.h5'))
            
            # Unfreeze layers for fine-tuning
            model = unfreeze_and_fine_tune(
                model,
                base_model,
                learning_rate=args.fine_tune_lr,
                unfreeze_layers=args.unfreeze_layers
            )
            
            t_start = time.time()
            fine_tune_history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=args.fine_tune_epochs,
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                callbacks=callbacks,
                class_weight=class_weights
            )
            t_end = time.time()
            print(f"Fine-tuning completed in {(t_end - t_start)/60:.2f} minutes")
            
            # Plot fine-tuning history
            plot_training_history(fine_tune_history, model_dir, "fine_tuning")
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            print("Continuing with best model from initial training.")
    
    # Load the best model weights
    if os.path.exists(os.path.join(model_dir, 'best_model.h5')):
        model.load_weights(os.path.join(model_dir, 'best_model.h5'))
    
    # Evaluate the final model
    print("\nEvaluating the final model...")
    evaluate_model(model, validation_generator, model_dir)
    
    # Save the model in TensorFlow SavedModel format
    final_model_path = os.path.join(model_dir, 'plant_classifier')
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Also save as HDF5 file for compatibility
    model.save(os.path.join(model_dir, 'plant_classifier.h5'))
    
    # Save model metadata
    with open(os.path.join(model_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Model: Plant Classifier (Layer 1)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: MobileNetV2 (transfer learning)\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Training data: {data_dir}\n")
        f.write(f"Dropout rate: {args.dropout_rate}\n")
        f.write(f"L2 regularization: {args.l2_reg}\n")
        f.write(f"Class distribution: {dict(zip(['not_plant', 'plant'], np.bincount(train_generator.classes)))}\n")
        f.write(f"Initial training epochs: {args.epochs}\n")
        if not args.no_fine_tune:
            f.write(f"Fine-tuning epochs: {args.fine_tune_epochs}\n")
            f.write(f"Unfrozen layers: {args.unfreeze_layers}\n")
        f.write(f"Final validation accuracy: {model.evaluate(validation_generator)[1]:.4f}\n")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()