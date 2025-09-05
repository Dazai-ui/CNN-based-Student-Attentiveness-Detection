# CNN-based Student Attentiveness Detection

## Overview

This project aims to develop a **Convolutional Neural Network (CNN)** model that can detect whether a student's eyes are open or closed, based on images. The model uses **data augmentation** techniques to improve its accuracy and generalization by creating more training data from the available images. The model is trained using both **original** and **augmented** images of students with open and closed eyes. The project leverages **TensorFlow** and **Keras** to train the CNN model and includes features like **PCA** for dimensionality reduction and evaluation metrics such as **accuracy**, **precision**, **recall**, and **F1 score**.

> This work is prepared as part of research contributions appearing in **Atlantic Press via Springer**.

---

## Dataset

The dataset used in this project is the **Drowsiness Detection Dataset** from Kaggle:

- [Drowsiness Detection Dataset on Kaggle](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset/data)

Dataset categories:

- **Open Eyes**  
- **Closed Eyes**

### Data Augmentation

- Rotation  
- Shifting  
- Zooming  
- Brightness adjustment  
- Horizontal flipping

### Preprocessing

- Resize all images to **64x64 pixels**  
- Normalize pixel values to `[0, 1]`

---

## Dimensionality Reduction with PCA

- **17 components** used to reduce feature dimensionality  
- Training features: `fit_transform()`  
- Test features: `transform()`  

---

## Model Architecture

### Feature Extractor (CNN)

- Conv2D layers for feature extraction  
- BatchNormalization and Dropout for regularization  
- MaxPooling layers  
- Flatten layer to convert 2D maps to 1D vectors  

### Classifier

- Dense layers with L2 regularization  
- Dropout layers  
- Sigmoid output for binary classification: Open Eyes / Closed Eyes  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix

---

## Training and Results

- 40 epochs, batch size = 8  
- ModelCheckpoint used to save best model  

### FLOPs Calculation

Estimated **Floating Point Operations (FLOPs)** for computational complexity.

---

## Files in the Repository

- **Cnn-based-student-attentiveness-detection.ipynb**: Full pipeline from data loading and augmentation to model training and evaluation.

---
## License

This project is **protected** and all rights are reserved by the author. See the `LICENSE` file for full details.  

- You may view, study, or use this project for **personal or academic purposes only**.  
- You may **NOT** redistribute, publish, sell, or sublicense this software or its contents without explicit permission from the author.  
- The software is provided “as is”, without any warranty of any kind.


