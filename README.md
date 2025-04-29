
# Lung Cancer Detection Web Application

This is a FastAPI-based web application for lung cancer detection using a HR-SEMobileCapsNet deep learning model. The application provides an intuitive user interface for uploading and analyzing lung images to detect normal, benign, or malignant conditions.

## Features

- **Intuitive UI**: Clean, responsive interface for easy interaction
- **Real-time Analysis**: Upload and analyze lung images instantly
- **Comprehensive Results**: Displays confidence scores for each class
- **Model Management**: Option to upload your trained model
- **Optimized Performance**: Fast inference with PyTorch
- **Dockerized Deployment**: Easy setup using Docker

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- FastAPI
- Docker (optional)

## Quick Start

### Option 1: Running with Python

1. Clone this repository:
   ```
   git clone <repository-url>
   cd lung-cancer-detection-app
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your trained model in the `models` directory with the name `final.pth`.
   If you don't have a trained model, you can upload it through the web interface later.

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:8000`

### Option 2: Running with Docker

1. Clone this repository:
   ```
   git clone <repository-url>
   cd lung-cancer-detection-app
   ```

2. Build and start the Docker container:
   ```
   docker-compose up -d
   ```

3. Open your browser and navigate to `http://localhost:8000`

## Model Architecture

The application uses a HR-SEMobileCapsNet model with the following components:

- **Feature Extractor**: Lightweight CNN with Squeeze-and-Excitation blocks
- **Primary Capsules**: Transform convolutional features into capsule representations
- **Digit Capsules**: Dynamic routing between capsules for classification

## Directory Structure

```
lung-cancer-detection-app/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── models/                # Directory for model files
│   └── final.pth          # Trained model (place your model here)
├── static/                # Static assets
├── templates/             # HTML templates
│   └── index.html         # Main UI template
└── uploads/               # Directory for uploaded images
```

## Using the Application

1. **Upload Image**: Click on the upload area or drag and drop your lung image file (JPG, JPEG, PNG)
2. **Analyze**: Click the "Analyze Image" button to process the image
3. **View Results**: See the classification result and confidence scores

If no model is loaded, you'll need to upload your trained PyTorch model file (.pth) first.

## Model Upload

If you need to upload a new model:

1. Click on the "Choose File" button in the model upload section
2. Select your .pth model file
3. Click "Upload Model"

## Technical Notes

- Images are preprocessed to grayscale, resized to 128x128, and normalized
- The model outputs three classes: Normal, Benign, and Malignant
- For best results, use high-quality lung CT scan images

## License

[Insert license information here]

## Acknowledgements

This application uses the HR-SEMobileCapsNet architecture for lung cancer detection.
>>>>>>> cc3ad5b (lung data model should be added)
