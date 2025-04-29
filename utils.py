import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os

def get_model_info():
    """Get information about the model architecture"""
    model_info = {
        "name": "HR-SEMobileCapsNet",
        "type": "Capsule Neural Network with Squeeze-and-Excitation",
        "input_size": "128x128 grayscale images",
        "classes": ["Normal", "Benign", "Malignant"],
        "parameters": "Approximately 1.2 million parameters",
        "feature_extractor": "3-layer CNN with SE blocks",
        "capsules": "Primary (8x8) and Digit (3x16) capsules"
    }
    return model_info

def preprocess_image(image):
    """Preprocess an image for the model"""
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 128x128
    image = image.resize((128, 128), Image.LANCZOS)
    
    return image

def create_heatmap(model, image_tensor, original_image):
    """Create a heatmap visualization of model attention"""
    model.eval()
    # Get the device
    device = next(model.parameters()).device
    
    # Ensure image tensor is on the correct device
    image_tensor = image_tensor.to(device)
    
    # Get feature maps (modify this according to your model architecture)
    with torch.no_grad():
        # Forward pass through feature extractor
        features = model.feature_extractor(image_tensor)
    
    # Get feature map activation (sum across channels)
    feature_map = features.squeeze(0).sum(dim=0).cpu().numpy()
    
    # Normalize for visualization
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    
    # Resize heatmap to original image size
    heatmap = Image.fromarray((feature_map * 255).astype(np.uint8)).resize(original_image.size, Image.LANCZOS)
    
    # Create a color heatmap
    heatmap_colored = np.zeros((heatmap.size[1], heatmap.size[0], 3), dtype=np.uint8)
    heatmap_array = np.array(heatmap)
    
    # Apply red color map (increase red channel relative to activation)
    heatmap_colored[:, :, 0] = heatmap_array  # Red
    
    # Convert to PIL Image
    heatmap_image = Image.fromarray(heatmap_colored)
    
    # Blend with original image
    original_rgb = original_image.convert('RGB')
    blended = Image.blend(original_rgb, heatmap_image, alpha=0.6)
    
    return blended

def generate_report(image_path, prediction, probabilities, class_labels):
    """Generate a detailed analysis report"""
    # Create a report with matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    img = Image.open(image_path)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Create bar chart for probabilities
    colors = ['green', 'orange', 'red']
    y_pos = np.arange(len(class_labels))
    
    bars = ax2.barh(y_pos, [p['value'] for p in probabilities], color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_labels)
    ax2.set_title("Classification Confidence")
    ax2.set_xlabel("Confidence (%)")
    
    # Add text annotations to bars
    for bar, prob in zip(bars, probabilities):
        ax2.text(min(bar.get_width() + 1, 95), 
                bar.get_y() + bar.get_height()/2, 
                f"{prob['value']:.1f}%", 
                va='center')
    
    # Add prediction result
    pred_color = {'Normal': 'green', 'Benign': 'orange', 'Malignant': 'red'}
    fig.suptitle(f"Prediction: {prediction}", 
                color=pred_color.get(prediction, 'black'),
                fontsize=16, fontweight='bold')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.02, 0.02, f"Generated: {timestamp}", fontsize=8)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

def save_upload_history(filename, prediction, probabilities):
    """Save upload and prediction history to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs('history', exist_ok=True)
    history_file = 'history/prediction_history.csv'
    
    # Create file with header if it doesn't exist
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            f.write("timestamp,filename,prediction,normal_prob,benign_prob,malignant_prob\n")
    
    # Add new entry
    with open(history_file, 'a') as f:
        probs = {p['label']: p['value'] for p in probabilities}
        f.write(f"{timestamp},{filename},{prediction},{probs.get('Normal', 0):.2f},"
                f"{probs.get('Benign', 0):.2f},{probs.get('Malignant', 0):.2f}\n")