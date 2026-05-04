
# Feature Extraction - Handwritten Digit Recognition

This branch focuses on extracting meaningful features from handwritten digit images to improve classification performance.

## Objective
To transform raw image data into structured numerical features suitable for machine learning models.

## Methods Used

### 1. Pixel Intensity
- Each image is represented using raw pixel values
- Images are flattened into 1D vectors
- Provides baseline feature representation

### 2. HOG (Histogram of Oriented Gradients)
- Extracts edge and gradient-based features
- Captures shape and structure of digits
- More informative than raw pixel values

## Process
- Load MNIST dataset using sklearn
- Normalize pixel values
- Apply HOG feature extraction
- Convert images into feature vectors

## Output
- Feature vectors representing each image
- Used as input for model training (SVM)

## Files
- 03_feature_extraction.ipynb → Feature extraction implementation

## Technologies Used
- Python
- NumPy
- Scikit-learn
- scikit-image (HOG)

## Contributors
- Anu Gopal V
- Anamika A
- Ardra vs 
