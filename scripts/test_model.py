# test_model.py
import os
import sys
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess an image for model prediction"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_path, img_size=(224, 224)):
    """Make prediction for a single image"""
    img_array = preprocess_image(image_path, img_size)
    if img_array is None:
        return None
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    is_plant = bool(prediction > 0.5)
    confidence = float(prediction) if is_plant else float(1.0 - prediction)
    
    return {
        "image_path": image_path,
        "is_plant": is_plant,
        "confidence": confidence,
        "prediction_value": float(prediction)
    }

def visualize_predictions(predictions, output_dir, num_samples=10):
    """Visualize some sample predictions"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly sample predictions if there are too many
    if len(predictions) > num_samples:
        samples = random.sample(predictions, num_samples)
    else:
        samples = predictions
    
    # Create a grid of images with their predictions
    rows = (len(samples) + 2) // 3  # 3 images per row
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if len(axes.shape) == 1 else [axes]
    
    for i, pred in enumerate(samples):
        if i >= len(axes):
            break
            
        try:
            # Load and display the image
            img = Image.open(pred["image_path"])
            axes[i].imshow(img)
            
            # Set title based on prediction
            title_color = 'green' if pred["is_plant"] else 'red'
            title = f"Plant ({pred['confidence']:.2%})" if pred["is_plant"] else f"Not Plant ({pred['confidence']:.2%})"
            axes[i].set_title(title, color=title_color)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error visualizing image {pred['image_path']}: {str(e)}")
            axes[i].set_title("Error loading image")
            axes[i].axis('off')
    
    # Hide unused axes
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, 'test_predictions.png')}")

def main():
    parser = argparse.ArgumentParser(description='Test the plant detection model on images')
    parser.add_argument('--model_path', type=str, default='models/layer1_is_plant/plant_classifier',
                        help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='models/layer1_is_plant/test_results',
                        help='Directory to save test results')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        help='Image dimensions (width, height)')
    parser.add_argument('--num_visualize', type=int, default=12,
                        help='Number of predictions to visualize')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    model_path = os.path.join(parent_dir, args.model_path)
    test_dir = os.path.join(parent_dir, args.test_dir)
    output_dir = os.path.join(parent_dir, args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        sys.exit(1)
    
    # Collect test images
    test_images = []
    if os.path.isdir(test_dir):
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    test_images.append(os.path.join(root, file))
    else:
        # If test_dir is actually a file
        if os.path.isfile(test_dir) and test_dir.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            test_images.append(test_dir)
    
    if not test_images:
        print(f"No test images found in {test_dir}")
        sys.exit(1)
    
    print(f"Found {len(test_images)} test images")
    
    # Make predictions
    all_predictions = []
    for image_path in tqdm(test_images):
        prediction = predict_image(model, image_path, tuple(args.img_size))
        if prediction is not None:
            all_predictions.append(prediction)
    
    # Calculate statistics
    num_plants = sum(1 for p in all_predictions if p["is_plant"])
    num_not_plants = len(all_predictions) - num_plants
    
    # Print summary
    print("\n===== Test Results =====")
    print(f"Total images: {len(all_predictions)}")
    print(f"Plants: {num_plants} ({num_plants/len(all_predictions):.2%})")
    print(f"Not plants: {num_not_plants} ({num_not_plants/len(all_predictions):.2%})")
    
    # Save predictions to JSON
    results = {
        "summary": {
            "total_images": len(all_predictions),
            "plants": num_plants,
            "not_plants": num_not_plants,
            "plant_percentage": num_plants/len(all_predictions) if all_predictions else 0
        },
        "predictions": all_predictions
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize some predictions
    visualize_predictions(all_predictions, output_dir, args.num_visualize)
    
    print(f"Test results saved to {output_dir}")

if __name__ == "__main__":
    main()