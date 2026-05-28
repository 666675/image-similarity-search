# Image Similarity Search Experiment
from google.colab import files
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Upload images
print(" Please upload at least 2 or 3 images for comparison:")
uploaded = files.upload()  # Opens image picker dialog

# Step 2: List uploaded files
image_paths = list(uploaded.keys())
print("✅ Uploaded files:", image_paths)

# Step 3: Pretrained CNN (ResNet50)
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)  # Updated method
model.eval()

# Step 4: Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_features(img_path):
    """Extract deep features from image"""
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(x)
    return features.flatten().numpy()

# Step 5: Extract features for all uploaded images
features = [get_features(path) for path in image_paths]

# Step 6: Compute pairwise similarities
sim_matrix = cosine_similarity(features)

# Step 7: Print results
print("\n Cosine Similarity Matrix (Higher = More Similar):")
print(sim_matrix)

# Step 8: Identify most similar pair
import numpy as np
np.fill_diagonal(sim_matrix, 0)
i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
print(f"\n✅ Most similar images are: {image_paths[i]} and {image_paths[j]}")