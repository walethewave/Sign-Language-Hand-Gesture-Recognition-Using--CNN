# Sign Language MNIST Classification Using CNN

![american_sign_language](https://github.com/user-attachments/assets/975aa9ec-920f-4041-959a-04ad329122b6)

## Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify American Sign Language (ASL) hand gestures from the Sign Language MNIST dataset. The dataset contains grayscale images of hand gestures representing the letters A-Z (excluding J and Z). The goal is to create a model that can accurately recognize and classify these gestures, which can be used in applications like real-time sign language translation, accessibility tools, and educational software.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Handling Overfitting](#handling-overfitting)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [Next Steps](#next-steps)
11. [How to Run the Code](#how-to-run-the-code)

## Introduction
The Sign Language MNIST dataset is derived from the American Sign Language (ASL) alphabet. It contains grayscale images of hand gestures representing the letters A-Z (excluding J and Z). Each image is 28x28 pixels, and the dataset is structured similarly to the MNIST dataset of handwritten digits.

This project aims to build a CNN to classify these hand gestures accurately. The model can be used in various applications, such as real-time sign language translation, accessibility tools, and educational software.

## Dataset
The dataset consists of two CSV files:
- `sign_mnist_train.csv`: Training data with 27,455 samples.
- `sign_mnist_test.csv`: Test data with 7,172 samples.

Each row in the dataset represents a 28x28 grayscale image. The first column is the label (0-24, representing letters A-Z excluding J and Z), and the remaining 784 columns represent pixel values (0-255).

## Exploratory Data Analysis (EDA)
- **Class Distribution**: Visualized the distribution of labels to check for class imbalance.
- **Pixel Intensity Distribution**: Analyzed the distribution of pixel intensities.
- **Statistical Measures**: Computed mean, variance, and standard deviation of pixel intensities.
- **Correlation Analysis**: Visualized correlations between pixel features using a heatmap.

## Data Preprocessing
- **Reshaping Images**: Reshaped the pixel data into 28x28x1 (height, width, channels).
- **Normalizing Pixel Values**: Normalized pixel values to the range [0, 1].
- **Splitting Data**: Split the dataset into training (80%) and validation (20%) sets.
- **Handling Label Issues**: Addressed the presence of label 0 by treating it as a valid class (25 classes in total).

## Model Architecture
The CNN architecture consists of:
- Two convolutional layers with ReLU activation.
- Two max-pooling layers.
- A fully connected layer with 128 units and ReLU activation.
- An output layer with 25 units (one for each class) and softmax activation.

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(25, activation='softmax')  # 25 classes (0-24)
])
