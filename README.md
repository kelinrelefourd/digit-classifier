# PyTorch Digit Classifier

A Convolutional Neural Network built with PyTorch to classify handwritten digits from the MNIST dataset.

## Project Overview
This project examines the basics of deep learning and computer vision. The goal was to create a model that recognizes handwritten digits (0-9) with high accuracy by using a standard CNN architecture.

The project shows:
- How to build a modular neural network in PyTorch.
- How to use Convolutional layers to extract spatial features.
- How to apply Regularization techniques like Dropout to avoid overfitting.
- How to evaluate model performance on unseen test data.

## Results
The model was trained for 5 epochs and tested on 10,000 images from the MNIST dataset.

- **Accuracy:** 99.17%
- **Average Loss:** 0.0242

## Project Structure
- `src/model.py`: Defines the CNN architecture (Input, Hidden Layers, Output).
- `src/train.py`: Manages data downloading, preprocessing, and the training loop.
- `src/eval.py`: Loads the saved model and checks performance on the test set.

## Installation and Usage

1. **Install Dependencies**
   To run this project, you will need the libraries in `requirements.txt`.
   ```bash
   pip install -r requirements.txt