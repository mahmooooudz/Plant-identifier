# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import random

def plot_training_history(history, output_dir, prefix=''):
    """Plot training history"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_training_history.png'))
    plt.close()
    
    # Also plot precision and recall if available
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_precision_recall.png'))
        plt.close()

def create_visual_samples(validation_generator, predictions, output_dir, num_samples=20):
    """Create visual samples of correct and incorrect predictions"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Get the true labels
    validation_generator.reset()
    y_true = validation_generator.classes
    
    # Convert predictions to binary classes
    y_pred_classes = (predictions > 0.5).astype(int).flatten()
    
    # Find correctly and incorrectly classified samples
    correct_indices = np.where(y_true == y_pred_classes)[0]
    incorrect_indices = np.where(y_true != y_pred_classes)[0]
    
    # Limit the number of samples
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples // 2, len(incorrect_indices))
    
    # Randomly sample from each group
    correct_samples = np.random.choice(correct_indices, num_correct, replace=False)
    incorrect_samples = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    
    # Get file paths and labels
    filenames = validation_generator.filenames
    class_indices = validation_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Function to save sample images
    def save_samples(indices, prefix):
        for i, idx in enumerate(indices):
            # Get the true and predicted labels
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred_classes[idx]]
            confidence = predictions[idx][0]
            
            # Get the file path
            file_path = os.path.join(validation_generator.directory, filenames[idx])
            
            # Load and save the image
            try:
                img = Image.open(file_path)
                save_path = os.path.join(output_dir, 'samples', f'{prefix}_{i}_{true_label}_as_{pred_label}_{confidence:.2f}.jpg')
                img.save(save_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Save sample images
    save_samples(correct_samples, 'correct')
    save_samples(incorrect_samples, 'incorrect')
    
    # Create a grid of sample images
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    # Function to display sample images in the grid
    def display_samples(indices, start_idx, title_color):
        for i, idx in enumerate(indices[:10]):  # Show up to 10 samples
            if i + start_idx >= len(axes):
                break
                
            # Get the true and predicted labels
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred_classes[idx]]
            confidence = predictions[idx][0]
            
            # Get the file path
            file_path = os.path.join(validation_generator.directory, filenames[idx])
            
            # Display the image
            try:
                img = Image.open(file_path)
                axes[i + start_idx].imshow(img)
                axes[i + start_idx].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", 
                                               color=title_color)
                axes[i + start_idx].axis('off')
            except Exception as e:
                print(f"Error displaying {file_path}: {str(e)}")
                axes[i + start_idx].set_title("Error loading image")
                axes[i + start_idx].axis('off')
    
    # Display samples
    display_samples(correct_samples, 0, 'green')
    display_samples(incorrect_samples, 10, 'red')
    
    # Hide any unused axes
    for i in range(min(10 + num_incorrect, 20), 20):
        axes[i].axis('off')
    
    plt.suptitle(f"Validation Samples: {num_correct} Correct (green) and {num_incorrect} Incorrect (red)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_samples.png'))
    plt.close()

def get_class_weights(y_train):
    """Calculate class weights for imbalanced datasets"""
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    n_classes = len(class_counts)
    
    class_weights = {}
    for i in range(n_classes):
        # Weight is inversely proportional to class frequency
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (n_classes * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    return class_weights