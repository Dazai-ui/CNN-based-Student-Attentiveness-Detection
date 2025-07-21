# CNN-based Student Attentiveness Detection

## Overview

This project aims to develop a **Convolutional Neural Network (CNN)** model that can detect whether a student's eyes are open or closed, based on images. The model uses **data augmentation** techniques to improve its accuracy and generalization by creating more training data from the available images. The model is trained using both **original** and **augmented** images of students with open and closed eyes. The project leverages **TensorFlow** and **Keras** to train the CNN model and includes features like **PCA** for dimensionality reduction and evaluation metrics such as **accuracy**, **precision**, **recall**, and **F1 score**.

## Dataset

The dataset consists of two main categories of images:
- **Open Eyes**
- **Closed Eyes**

These images were used to train the model to recognize whether a student’s eyes are open or closed.

### **Data Augmentation:**
To improve the model's performance and avoid overfitting, aggressive data augmentation techniques were applied, including:
- Rotation
- Shifting
- Zooming
- Brightness adjustment
- Horizontal flipping

### **Preprocessing:**
- All images were resized to **64x64 pixels**.
- The pixel values were normalized to the range `[0, 1]`.

## Dimensionality Reduction with PCA

To reduce the feature space and prevent overfitting, **Principal Component Analysis (PCA)** is applied to the features extracted by the CNN model. This process reduces the number of features while retaining the most important components that explain the variance in the data.

- **PCA with 17 components** is used to transform the high-dimensional feature vectors into a lower-dimensional space before passing them to the classifier.
- This dimensionality reduction helps improve model performance and speeds up training and inference.

### **PCA Application:**
- **`PCA(n_components=17)`**: Reduces the feature vectors to **17 components** that capture the most variance in the data.
- **Training features** are transformed using **`fit_transform`**, while **test features** are transformed using **`transform`**.

## Model Architecture

### **Feature Extractor Model:**
The CNN model consists of the following layers:
- **Conv2D** layers to extract features from the images.
- **BatchNormalization** and **Dropout** layers to prevent overfitting.
- **MaxPooling** layers to reduce dimensionality.
- **Flatten** layer to convert the 2D feature maps into a 1D vector.

### **Classifier Model:**
The model is followed by a **fully connected neural network** (Dense layers) with:
- **L2 regularization** to avoid overfitting.
- **Dropout** to randomly disable neurons during training.

The output layer uses a **sigmoid** activation function to classify the images into two categories:
- Open Eyes
- Closed Eyes

## Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Additionally, a **confusion matrix** is used to visualize the true positive, false positive, true negative, and false negative predictions.

## Training and Results

The model is trained for **40 epochs** with a **batch size of 8**, and a **ModelCheckpoint** callback is used to save the best model based on validation loss.

### **FLOPs Calculation**:
An estimation of the **Floating Point Operations (FLOPs)** for the model is done to determine its computational complexity, though full FLOPs profiling might require deeper logging analysis.

## Files in the Repository

- **Cnn-based-student-attentiveness-detection.ipynb**: The main notebook (suitable for Jupyter, Google Colab, Kaggle etc) containing the entire pipeline, from data loading and augmentation to model training and evaluation.
