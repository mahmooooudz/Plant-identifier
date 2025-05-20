# plant_identifier.py

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import traceback
import json
import time

# Configuration
MODEL_DIR = "models/layer1_is_plant"  # Directory where models are saved
IMG_SIZE = (224, 224)                 # Input image size

# Define plant-related keywords for classification (used by fallback classifier)
PLANT_KEYWORDS = [
    # General plant terms
    'plant', 'flower', 'tree', 'bush', 'shrub', 'herb', 'grass',
    
    # Specific plant types
    'rose', 'daisy', 'sunflower', 'orchid', 'tulip', 'lily', 'dandelion',
    'maple', 'oak', 'pine', 'fern', 'moss', 'cactus', 'palm', 'bamboo',
    
    # Plant parts
    'leaf', 'leaves', 'stem', 'root', 'trunk', 'branch', 'twig',
    'petal', 'blossom', 'bloom', 'bud', 'seed', 'fruit', 'berry',
    
    # Agricultural plants
    'corn', 'wheat', 'rice', 'soybean', 'barley', 'oat', 'cotton',
    'vegetable', 'potato', 'tomato', 'carrot', 'lettuce', 'cabbage',
    'broccoli', 'spinach', 'onion', 'garlic', 'cucumber', 'zucchini',
    
    # Fruits
    'apple', 'orange', 'banana', 'grape', 'strawberry', 'blueberry',
    'raspberry', 'blackberry', 'cherry', 'peach', 'pear', 'plum',
    'watermelon', 'melon', 'pineapple', 'mango', 'kiwi', 'lemon', 'lime',
    
    # Environments
    'garden', 'forest', 'jungle', 'orchard', 'vineyard', 'meadow',
    
    # Special plant classifications
    'cardoon', 'buckeye', 'megalith', 'hay', 'lakeside', 'paddy', 'pitcher_plant', 'acorn',
    'fig', 'pitcher', 'carnivorous'
]

# Define plant container keywords
PLANT_CONTAINERS = [
    'pot', 'vase', 'planter', 'container', 'flowerpot', 'jardiniere', 
    'urn', 'basket', 'window_box', 'windowbox', 'trough', 'terrarium', 
    'greenhouse', 'ceramic_pot', 'clay_pot', 'plant_pot', 'terracotta',
    'houseplant', 'indoor_plant', 'potted_plant', 'potted', 'flower_pot'
]

# Absolute non-plant categories that should NEVER be classified as plants
ABSOLUTE_NON_PLANTS = [
    # Street elements
    'street_sign', 'stop_sign', 'traffic_light', 'signboard', 'sign', 
    'traffic_sign', 'billboard', 'parking_meter', 'pole',
    
    # People
    'person', 'human', 'man', 'woman', 'boy', 'girl', 'child', 'face',
    
    # Clothing 
    'lab_coat', 'jersey', 'gown', 'dress', 'maillot', 'suit', 'tuxedo',
    'clothing', 'uniform', 'coat', 'tie', 'bow_tie', 'necktie',
    
    # Electronics and devices
    'computer', 'laptop', 'smartphone', 'phone', 'tablet', 'keyboard', 
    'mouse', 'monitor', 'television', 'tv', 'remote', 'camera', 'microphone',
    
    # Other common false positives
    'abacus', 'balloon', 'ball', 'book', 'doorknob', 'weapon', 'gun',
    'mask', 'toy', 'panda', 'umbrella', 'wallet', 'wristwatch', 'watch'
]

# Define non-plant categories (used by fallback classifier)
NON_PLANT_CATEGORIES = [
    # Vehicles and transportation
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'ship', 'speedboat', 'sports_car',
    'race_car', 'racing_car', 'convertible', 'minivan', 'jeep', 'limousine', 'ambulance', 
    'fire_engine', 'tire', 'wheel', 'car_wheel', 'steering_wheel', 'airliner', 'airplane',
    
    # Household items (excluding containers that might hold plants)
    'chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'lamp', 'clock', 'watch',
    'iron', 'vacuum', 'broom', 'mop', 'oven', 'microwave', 'refrigerator', 'fridge',
    'clog', 'wooden_spoon', 'plate', 'cup', 'mug', 'fork', 'knife', 'spoon',
    
    # Animals
    'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'sheep', 'pig', 'animal', 'insect'
]

# Ambiguous terms that could be plants or non-plants
AMBIGUOUS_TERMS = {
    # Term: confidence threshold to consider as plant
    'pitcher': 0.4,  # Pitcher plant vs. water pitcher
    'fig': 0.5,      # Fig fruit vs. figure
    'vase': 0.4      # Vase with plant vs. empty vase
}

# Minimum confidence threshold for plant classification
MIN_PLANT_CONFIDENCE = 0.10  # 10% confidence required for plant classification

# For TensorFlow 2.x compatibility and performance optimization
try:
    # Enable mixed precision for faster performance
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled for better performance")
    
    # Enable memory growth on GPUs if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU memory growth enabled for {len(physical_devices)} devices")
    else:
        print("No GPU found, using CPU for inference")
        
    # Set inter and intra parallelism threads
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
except Exception as e:
    print(f"Warning: Performance optimization failed: {e}")

class PlantIdentifier:
    """
    First layer of plant identification system:
    Determines whether an image contains a plant or not.
    """
    
    def __init__(self, model_path=None):
        """Initialize the plant identifier model"""
        # Set model path
        if model_path is None:
            self.model_path = os.path.join(MODEL_DIR, "plant_classifier")
            self.fallback_model_path = os.path.join(MODEL_DIR, "mobilenet_v2_model")
        else:
            self.model_path = model_path
            self.fallback_model_path = model_path + "_fallback"
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try to load the fine-tuned model
        self.is_fine_tuned = False
        start_time = time.time()
        
        try:
            # First try loading the SavedModel format
            if os.path.isdir(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.is_fine_tuned = True
                print(f"Loaded fine-tuned plant classifier model from {self.model_path}")
            # Then try loading the HDF5 format
            elif os.path.isfile(self.model_path + ".h5"):
                self.model = tf.keras.models.load_model(self.model_path + ".h5")
                self.is_fine_tuned = True
                print(f"Loaded fine-tuned plant classifier model from {self.model_path}.h5")
            # If neither exists, fall back to MobileNetV2
            else:
                print(f"Fine-tuned model not found. Using MobileNetV2 fallback.")
                self._create_fallback_model()
            
            # Warm up the model with a dummy prediction
            dummy_input = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
            
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating fallback model...")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback MobileNetV2 model if fine-tuned model is not available"""
        try:
            # First try to load a saved fallback model
            if os.path.exists(self.fallback_model_path):
                self.model = tf.keras.models.load_model(self.fallback_model_path)
                print(f"Loaded fallback model from {self.fallback_model_path}")
            else:
                # Load a pre-trained MobileNetV2 model
                self.model = tf.keras.applications.MobileNetV2(
                    weights='imagenet',
                    include_top=True,
                    input_shape=(*IMG_SIZE, 3)
                )
                print("Created new MobileNetV2 fallback model")
                
                # Save the model for future use
                try:
                    self.model.save(self.fallback_model_path)
                    print(f"Fallback model saved to {self.fallback_model_path}")
                except Exception as e:
                    print(f"Error saving fallback model: {e}")
            
            self.is_fine_tuned = False
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Warm up the model
            dummy_input = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
            
        except Exception as e:
            print(f"Critical error creating fallback model: {e}")
            print(traceback.format_exc())
            raise
    
    def _preprocess_image(self, image_path):
        """
        Preprocess an image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize the image
            img = img.resize(IMG_SIZE)
            
            # Convert to array and add batch dimension
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply appropriate preprocessing
            if self.is_fine_tuned:
                # For fine-tuned model, just normalize to [0,1]
                return img_array.astype(np.float32) / 255.0
            else:
                # For MobileNetV2 with ImageNet weights
                return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            print(traceback.format_exc())
            return None
    
    def predict(self, image_path):
        """
        Predict whether an image contains a plant or not
        
        Args:
            image_path: Path to the image file
            
        Returns:
            A dictionary containing prediction results
        """
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                "error": f"Image not found: {image_path}",
                "is_plant": False,
                "confidence": 0.0
            }
        
        # Preprocess the image
        img_array = self._preprocess_image(image_path)
        if img_array is None:
            return {
                "error": "Failed to process image",
                "is_plant": False,
                "confidence": 0.0
            }
        
        try:
            # Make prediction based on model type
            if self.is_fine_tuned:
                # Binary classification with fine-tuned model
                prediction = self.model.predict(img_array, verbose=0)[0][0]
                is_plant = bool(prediction > 0.5)
                confidence = float(prediction) if is_plant else float(1.0 - prediction)
                
                return {
                    "is_plant": is_plant,
                    "confidence": confidence,
                    "model_type": "fine_tuned"
                }
            else:
                # Using the fallback ImageNet classifier with keyword matching
                predictions = self.model.predict(img_array, verbose=0)
                decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]
                
                # Collect all predictions with scores for analysis
                all_predictions = [(label.lower(), float(score)) for _, label, score in decoded_predictions]
                
                # Check for absolute non-plants with high confidence
                is_blocked = False
                for _, label, score in decoded_predictions[:3]:
                    label_lower = label.lower()
                    if any(blocker in label_lower for blocker in ABSOLUTE_NON_PLANTS) and float(score) > 0.15:
                        is_blocked = True
                        break
                
                # If blocked, immediately classify as non-plant
                if is_blocked:
                    return {
                        "is_plant": False,
                        "confidence": 1.0 - float(decoded_predictions[0][2]),
                        "top_predictions": [
                            {"label": label, "confidence": float(score)} 
                            for _, label, score in decoded_predictions[:5]
                        ],
                        "model_type": "fallback"
                    }
                
                # Check for plant-related predictions
                plant_predictions = []
                for _, label, score in decoded_predictions:
                    label_lower = label.lower()
                    
                    # Skip if label is in our explicit non-plant list
                    if any(non_plant in label_lower for non_plant in NON_PLANT_CATEGORIES):
                        continue
                    
                    # Check if label contains any of our plant keywords
                    if any(keyword in label_lower for keyword in PLANT_KEYWORDS) or \
                       any(container in label_lower for container in PLANT_CONTAINERS):
                        plant_predictions.append((label, float(score)))
                
                # Check for plant containers
                has_plant_container = False
                for label, score in all_predictions[:5]:
                    if any(container in label for container in PLANT_CONTAINERS) and score > 0.15:
                        has_plant_container = True
                        if not plant_predictions:
                            plant_predictions = [(decoded_predictions[0][1], float(decoded_predictions[0][2]))]
                        break
                
                # Determine if it's a plant
                is_plant = len(plant_predictions) > 0 or has_plant_container
                
                # Enforce minimum confidence threshold
                if is_plant and plant_predictions and plant_predictions[0][1] < MIN_PLANT_CONFIDENCE:
                    is_plant = False
                    plant_predictions = []
                
                # Prepare result
                if is_plant:
                    return {
                        "is_plant": True,
                        "confidence": float(plant_predictions[0][1]),
                        "plant_type": plant_predictions[0][0],
                        "top_predictions": [
                            {"label": label, "confidence": float(score)} 
                            for _, label, score in decoded_predictions[:5]
                        ],
                        "model_type": "fallback"
                    }
                else:
                    return {
                        "is_plant": False,
                        "confidence": 1.0 - float(decoded_predictions[0][2]) if decoded_predictions else 0.5,
                        "top_predictions": [
                            {"label": label, "confidence": float(score)} 
                            for _, label, score in decoded_predictions[:5]
                        ],
                        "model_type": "fallback"
                    }
                    
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            print(error_message)
            print(traceback.format_exc())
            return {
                "error": error_message,
                "is_plant": False,
                "confidence": 0.0
            }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Plant Identification - Layer 1: Is it a plant?')
    parser.add_argument('--image', type=str, required=True, help='Path to the image for analysis')
    parser.add_argument('--model', type=str, default=None, help='Path to the model (optional)')
    parser.add_argument('--debug', action='store_true', help='Show detailed prediction information')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Create the plant identifier
    identifier = PlantIdentifier(model_path=args.model)
    
    # Make prediction
    result = identifier.predict(args.image)
    
    # Display results
    if args.json:
        # Print as JSON
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n===== Plant Analysis Results =====")
            model_type = result.get("model_type", "unknown")
            if result["is_plant"]:
                print(f"This is a plant!")
                if "plant_type" in result:
                    print(f"Identified as: {result['plant_type']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Model type: {model_type}")
            else:
                print(f"This is not a plant")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Model type: {model_type}")
            
            if "top_predictions" in result and args.debug:
                print("\nTop predictions:")
                for i, pred in enumerate(result["top_predictions"][:5], 1):
                    print(f"{i}. {pred['label']} ({pred['confidence']:.2%})")

if __name__ == "__main__":
    main()