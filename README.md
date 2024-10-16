# Plant Disease Detection Application

![Project Image](link-to-image)

## Project Overview

The Plant Disease Detection application identifies diseases or anomalies in plants based on images of their leaves. This project aims to create a comprehensive machine learning pipeline for image classification, facilitating the detection of various plant diseases.

## Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **MLOps**: For deploying the application

## Functionalities

### 1. Data Preparation and Preprocessing
The initial phase involves preparing and preprocessing the dataset. Key tasks include:

- Resizing images to a uniform dimension.
- Rescaling pixel values to a range between 0 and 1.
- Applying data augmentation to artificially expand the dataset through random transformations such as flipping, rotating, and zooming.
- Optimizing the TensorFlow pipeline using techniques like caching and prefetching to enhance training performance.

### 2. Training the Deep Learning Model
The model employs a Convolutional Neural Network (CNN) built using TensorFlow's Keras API. Key components of the model include:

- Six `Conv2D` layers, each followed by `MaxPooling2D` to reduce spatial dimensions while capturing essential features.
- ReLU activation functions.
- A Dense layer with 64 units, culminating in a softmax output layer for multi-class classification (15 classes).
- Compiled with the Adam optimizer and Sparse Categorical Crossentropy, and trained using `model.fit()`, with accuracy as the evaluation metric.

### 3. Application Development and Deployment
In this phase, we can make predictions and save the model as a `.keras` file for future use. The model is integrated into the Flask application to enable prediction capabilities. 

- **Containerization**: The application is containerized using Docker to ensure consistent deployment across various environments, simplifying dependency management and enhancing scalability.

