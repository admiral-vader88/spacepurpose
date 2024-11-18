from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Debugging working directory
print("Current working directory:", os.getcwd())

# Flask app initialization
app = Flask(__name__)

# Load Model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    # Render index.html from the templates directory
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    label = 'non_recyclable' if predicted.item() == 0 else 'recyclable'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)

