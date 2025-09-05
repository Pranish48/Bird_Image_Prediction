import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_PATH = "best_bird_model.pth"
num_classes = 200

# Build ResNet50 architecture
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load trained model
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

df = pd.read_csv("bird_metadata.csv")

# Create a dictionary: 0-based class index → class name
class_dict = dict(zip(df['class_id'] - 1, df['class_name']))


def predict_image(image):
    x = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        _, pred_idx = torch.max(outputs, 1)
        return class_dict[pred_idx.item()]


# Streamlit User Interface

st.title("🐦 Bird Species Predictor")
st.write("Upload an image of a bird and the model will predict its species.")
st.write("Works well with North American Birds -CUB-200-2011")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    st.write("Predicting...")
    pred_name = predict_image(image)
    
    # Show prediction
    clean_name = pred_name.split('.', 1)[-1].replace('_', ' ')
    st.success(f"Predicted Bird Species: {clean_name}")
