# app.py
import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms
import uvicorn

# Define the same model architecture from the provided code
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            SEBlock(16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            SEBlock(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            SEBlock(64)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, capsule_dim, num_capsules, kernel_size=3, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=kernel_size, stride=stride, padding=1)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x)
        u = u.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        u = u.permute(0, 1, 3, 2).contiguous()
        u = u.view(batch_size, -1, self.capsule_dim)
        return self.squash(u)

    def squash(self, s):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        return (s_norm**2 / (1 + s_norm**2)) * (s / (s_norm + 1e-8))

class DigitCapsules(nn.Module):
    def __init__(self, input_capsules, input_dim, num_classes, capsule_dim, routing_iters=3):
        super(DigitCapsules, self).__init__()
        self.num_classes = num_classes
        self.routing_iters = routing_iters
        self.input_capsules = input_capsules
        self.W = nn.Parameter(0.01 * torch.randn(1, input_capsules, num_classes, capsule_dim, input_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)  # [batch_size, input_capsules, 1, input_dim, 1]
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)  # [batch_size, input_capsules, num_classes, capsule_dim]
        
        b_ij = torch.zeros(batch_size, self.input_capsules, self.num_classes, 1, device=x.device)
        
        for _ in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)
            
        v_j = v_j.squeeze(1)  # [batch_size, num_classes, capsule_dim]
        return v_j

    def squash(self, s):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        return (s_norm**2 / (1 + s_norm**2)) * (s / (s_norm + 1e-8))

class HR_SEMobileCapsNet(nn.Module):
    def __init__(self, num_classes=3):
        super(HR_SEMobileCapsNet, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.primary_capsules = PrimaryCapsules(
            in_channels=64,
            capsule_dim=8,
            num_capsules=8,
            kernel_size=3,
            stride=2
        )
        self.digit_capsules = DigitCapsules(
            input_capsules=8*16*16,
            input_dim=8,
            num_classes=num_classes,
            capsule_dim=16
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return torch.norm(x, dim=-1)  # Output vector norms as class probabilities

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Lung Cancer Detection System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize model and set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HR_SEMobileCapsNet(num_classes=3).to(device)

# Try to load the model
try:
    model.load_state_dict(torch.load("models/final.pth", map_location=device))
    model_loaded = True
except:
    model_loaded = False
    print("Model file not found. Place your trained model at 'models/final.pth'")

model.eval()

# Class labels
class_labels = ['Normal', 'Benign', 'Malignant']

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_loaded": model_loaded})

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    if not model_loaded:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "Model not loaded. Please upload model file first.", "model_loaded": model_loaded}
        )
    
    # Validate file format
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "Only PNG, JPG, and JPEG files are allowed.", "model_loaded": model_loaded}
        )
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save original image
        img_path = f"uploads/{file.filename}"
        with open(img_path, "wb") as f:
            f.write(contents)
            
        # Transform image for model input
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        # Convert probabilities to percentage
        probs = probabilities.squeeze().cpu().numpy() * 100
        
        # Create base64 image for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Return result
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "model_loaded": model_loaded,
                "img_data": f"data:image/jpeg;base64,{img_str}",
                "prediction": class_labels[predicted_class],
                "probabilities": [
                    {"label": class_labels[i], "value": float(probs[i])} 
                    for i in range(len(class_labels))
                ],
                "filename": file.filename
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"Error processing image: {str(e)}", "model_loaded": model_loaded}
        )

@app.post("/upload_model/")
async def upload_model(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pth'):
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "Only .pth model files are allowed.", "model_loaded": model_loaded}
        )
    
    try:
        contents = await file.read()
        with open("models/final.pth", "wb") as f:
            f.write(contents)
            
        # Load the model
        global model, model_loaded
        model.load_state_dict(torch.load("models/final.pth", map_location=device))
        model.eval()
        model_loaded = True
        
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "success": "Model successfully uploaded and loaded!", "model_loaded": model_loaded}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"Error loading model: {str(e)}", "model_loaded": model_loaded}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)