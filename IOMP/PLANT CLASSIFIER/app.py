# app.py
from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64

# -------------------------
# Config / constants
# -------------------------
IMG_SIZE = 224
RESIZE_TO = int(IMG_SIZE * 1.1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# exact transforms
valid_transforms = transforms.Compose([
    transforms.Resize(RESIZE_TO),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Correct plant disease classes (exact folder order)
# -------------------------
classes = [
    "Apple_Cedar_apple_rust",
    "Apple_healthy",
    "Cherry_(including_sour)_healthy",
    "Corn_(maize)_healthy",
    "Grape_Black_rot",
    "Grape_healthy",
    "Orange_Haunglongbing_(Citrus_greening)",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Squash_Powdery_mildew",
    "Strawberry_Leaf_scorch",
    "Strawberry_healthy",
    "Tomato_Leaf_Mold",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -------------------------
# Load model
# -------------------------
def load_model(model_path_candidates=None):
    if model_path_candidates is None:
        model_path_candidates = ['./best_model.pth']

    loaded = False
    model = None

    for path in model_path_candidates:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

                num_classes = state_dict['fc.weight'].shape[0]
                if num_classes != len(classes):
                    raise ValueError(f"Checkpoint has {num_classes} classes, but classes list has {len(classes)}")

                # create ResNet50
                model = models.resnet50(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)

                model.load_state_dict(state_dict)
                loaded = True
                print(f"Loaded model from {path} with {num_classes} classes")
                break

            except Exception as e:
                print(f"Failed loading from {path}: {e}")

    if not loaded:
        raise FileNotFoundError(f"Could not load model. Tried: {model_path_candidates}")

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder="static")


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🌱 Plant Disease Detection</title>
<style>
 body { 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    background-image: url('/static/452600.jpg');
    background-size: cover;      /* cover the entire page */
    background-position: center; /* center the image */
    background-repeat: no-repeat;
    color: #333; 
    text-align: center; 
    padding: 50px; 
}
  h1 { 
    color: white;      /* changed from #00796b to white */
    margin-bottom: 20px; 
    font-size: 2.5em; 
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7); /* optional: adds shadow for readability */
}

  form { 
    margin-bottom: 30px; 
    background: #ffffffaa; 
    padding: 30px; 
    border-radius: 15px; 
    box-shadow: 0px 5px 15px rgba(0,0,0,0.2); 
    display: inline-block; 
  }
  input[type=file] { 
    padding: 10px; 
    margin: 10px 0; 
    border-radius: 8px; 
    border: 1px solid #00796b; 
  }
  input[type=submit] { 
    background-color: #00796b; 
    color: white; 
    padding: 12px 25px; 
    border: none; 
    border-radius: 10px; 
    cursor: pointer; 
    font-size: 1em; 
    transition: background 0.3s ease; 
  }
  input[type=submit]:hover { 
    background-color: #004d40; 
  }

  .result {
    background-color: #ffffffaa;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
}

  img.uploaded-img { 
    max-width: 300px; 
    margin-top: 20px; 
    border-radius: 15px; 
    box-shadow: 0px 5px 15px rgba(0,0,0,0.2); 
  }
  .prediction-text { 
    margin-top: 15px; 
    line-height: 1.6em; 
  }
  .prediction-label {
    font-family: 'Times New Roman', Times, serif; 
    font-weight: normal; 
    font-size: 1.2em;
  }
  .prediction-disease {
    font-family: 'Times New Roman', Times, serif; 
    font-weight: bold; 
    font-size: 1.4em; 
    margin-bottom: 10px;
  }
  .prediction-confidence {
    font-family: 'Times New Roman', Times, serif; 
    font-weight: normal; 
    font-size: 1.2em;
  }
</style>
</head>
<body>
<h1>🌱 Plant Disease Detection</h1>
<form method="post" enctype="multipart/form-data" action="/predict">
  <input type="file" name="file" accept="image/*" required>
  <br>
  <input type="submit" value="Upload & Predict">
</form>

{% if prediction %}
<div class="result">
  <img src="data:image/png;base64,{{ prediction['image'] }}" class="uploaded-img">
  <div class="prediction-text">
    <div class="prediction-label">Predicted Disease:</div>
    <div class="prediction-disease">{{ prediction['class'] }}</div>
    <div class="prediction-confidence">Confidence: {{ prediction['confidence'] }}</div>
  </div>
</div>
{% endif %}
</body>
</html>
"""



# -------------------------
# Helper functions
# -------------------------
def allowed_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return valid_transforms(image).unsqueeze(0)

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# -------------------------
# Routes
# -------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template_string(HTML_PAGE, prediction={'class': 'No file uploaded', 'confidence': 0, 'image': ''})

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template_string(HTML_PAGE, prediction={'class': 'Invalid file', 'confidence': 0, 'image': ''})

    try:
        img_bytes = file.read()
        input_tensor = transform_image(img_bytes).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_idx = pred_idx.item()
            conf = round(conf.item(), 4)

        predicted_class = classes[pred_idx]
        img_b64 = image_to_base64(img_bytes)

        prediction = {
            'class': predicted_class,
            'confidence': conf,
            'image': img_b64
        }

        return render_template_string(HTML_PAGE, prediction=prediction)

    except Exception as e:
        return render_template_string(HTML_PAGE, prediction={'class': f'Error: {str(e)}', 'confidence': 0, 'image': ''})

# -------------------------
# Run Flask
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
