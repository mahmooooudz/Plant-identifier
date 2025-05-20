# Plant Identification System

A comprehensive plant identification system with a multi-layer approach to accurately identify plants from images.

## Overview

This system uses a funnel approach to plant identification:

1. **Layer 1**: Is it a plant? (Binary classification)
2. **Layer 2**: What type of plant? (Tree, flower, grass, etc.)
3. **Layer 3**: What plant part? (Leaf, flower, fruit, etc.)
4. **Layer 4**: What genus?
5. **Layer 5**: What species?

The current implementation focuses on Layer 1, which distinguishes plants from non-plants with high accuracy.

## Features

- Web interface for easy image uploads
- Batch processing capability
- Fine-tuned machine learning model for plant detection
- Detailed visual results with confidence scores
- Command-line interface for scripting
- Dataset downloader and model training scripts

## Installation

### Prerequisites

- Windows operating system
- Python 3.7 or later
- Internet connection (for downloading datasets and dependencies)

### Setup

1. Clone this repository or extract the project files
2. Run `install_dependencies.bat` to install required Python packages
3. Download training datasets with `run_data_download.bat` (optional if you want to train your own model)
4. Train the model using `run_training.bat` (optional - a pre-trained model will be downloaded automatically if needed)
5. Start the web application with `run_web_app.bat`

## Usage

### Web Interface

1. Open your web browser and go to `http://localhost:8000`
2. Upload an image and click "Analyze Image"
3. View the results showing whether the image contains a plant

### Command Line

For individual image analysis:
```
python plant_identifier.py --image path/to/image.jpg
```

Additional options:
```
python plant_identifier.py --image path/to/image.jpg --debug --json
```

## Training Your Own Model

To train your own model:

1. Download datasets using `run_data_download.bat`
2. Run the training script with `run_training.bat`
3. Monitor the training progress in the console
4. The trained model will be saved in the `models/layer1_is_plant` directory

## Project Structure

```
plant_identifier/
├── data/                         # Training data and datasets
│   ├── download_datasets.py      # Script to download training data
│   └── dataset_urls.json         # Configuration for dataset downloads
├── models/                       # Trained models
│   └── layer1_is_plant/          # Plant vs. non-plant model
├── scripts/                      # Training and testing scripts
│   ├── train_layer1.py           # Training script for layer 1
│   ├── test_model.py             # Script for testing the model
│   └── utils.py                  # Utility functions
├── web_app/                      # Web interface
│   ├── static/                   # Static files
│   ├── templates/                # HTML templates
│   └── app.py                    # FastAPI web application
├── plant_identifier.py           # Main plant identifier module
├── install_dependencies.bat      # Script to install dependencies
├── run_web_app.bat               # Script to start the web app
├── run_data_download.bat         # Script to download datasets
├── run_training.bat              # Script to train the model
└── README.md                     # Project documentation
```

## Next Steps

Future development will focus on implementing the remaining layers:
- Layer 2: Plant type classification
- Layer 3: Plant part identification
- Layer 4: Genus classification
- Layer 5: Species identification with GBIF integration

## Troubleshooting

- **Error loading model**: The system will fall back to a pre-trained MobileNetV2 model
- **Slow processing**: Consider using a smaller batch size or installing TensorFlow with GPU support
- **Missing dependencies**: Run `install_dependencies.bat` again
- **Dataset download issues**: Check your internet connection and firewall settings