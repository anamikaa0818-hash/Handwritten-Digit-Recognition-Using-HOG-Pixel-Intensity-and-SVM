# Handwritten-Digit-Recognition-Using-HOG-Pixel-Intensity-and-SVM

# Problem Definition 

This branch focuses on defining the problem statement and understanding the initial setup for handwritten digit recognition using machine learning.

## Objective
To clearly define the problem of recognizing handwritten digits and establish the foundation for building a machine learning model.

## Problem Statement
Handwritten digit recognition is a classification problem where the goal is to correctly identify digits (0–9) from image data. This problem is widely used in applications such as postal code recognition, bank cheque processing, and digitizing handwritten documents.

## Dataset Overview
- Dataset: MNIST
- Source: OpenML / Scikit-learn
- Total samples: 70,000 images
- Image size: 28 × 28 pixels
- Classes: 10 (digits 0–9)

## Initial Analysis
- Loaded dataset using sklearn
- Inspected data structure and shape
- Observed pixel values and labels
- Verified class distribution

## Key Observations
- Each image is represented as numerical pixel values
- Pixel intensity ranges from 0 to 255
- Dataset contains balanced classes
- Data is suitable for classification tasks

## Scope of the Project
- Apply preprocessing and feature extraction
- Train machine learning models (SVM)
- Evaluate model performance
- Deploy the model using Streamlit

## Files
- 01_problem_definition.ipynb → Initial dataset loading and understanding

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

## Contributors
- Anu Gopal V
- Anamika A
- Ardra vs 
