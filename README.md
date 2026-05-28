# Image Similarity Search using Deep Learning

This project performs Image Similarity Search using a pretrained ResNet50 CNN model and calculates similarity between uploaded images using Cosine Similarity.

The system extracts deep features from images and identifies which images are most visually similar.

## Features

* Upload multiple images
* Extract deep features using ResNet50
* Compute cosine similarity between images
* Display similarity matrix
* Identify the most similar image pair
* Uses PyTorch pretrained models

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* PIL (Python Imaging Library)

## Installation

Install required libraries:

```bash
pip install torch torchvision pillow numpy scikit-learn
```

## How to Run

```bash
python image_similarity.py
```

Or run directly in Google Colab.

## Project Workflow

1. Upload images
2. Preprocess images
3. Extract features using ResNet50
4. Compute cosine similarity
5. Compare image similarity scores
6. Display most similar images

## Sample Output

```bash
Cosine Similarity Matrix:
[[1.00 0.87 0.62]
 [0.87 1.00 0.59]
 [0.62 0.59 1.00]]

Most similar images are:
image1.jpg and image2.jpg
```

## Project Structure

```bash
Image-Similarity-Search/
│
├── image_similarity.py
├── README.md
└── requirements.txt
```

## Future Improvements

* Build web interface using Streamlit
* Support large datasets
* Faster image retrieval system
* Add image search functionality
