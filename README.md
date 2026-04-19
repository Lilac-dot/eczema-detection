# Multi-Modal Eczema Detection and Monitoring System

## Overview

This project presents a machine learning-based approach for detecting and monitoring eczema using multiple data modalities. The system combines image-based classification with sensor-based models to capture both visible skin conditions and underlying physiological and behavioral factors.

The primary component is a Convolutional Neural Network (CNN) trained to classify skin images as eczema-affected or normal. In addition, separate machine learning models are designed to estimate stress levels and detect scratching behavior, both of which are known contributors to eczema flare-ups.

---

## Objectives

The main objectives of this project are:

* To classify skin images into eczema and normal categories using deep learning
* To model physiological signals such as electrodermal activity and temperature for stress detection
* To infer scratching behavior from motion-related data
* To combine these components into a unified framework for estimating eczema flare risk

---

## System Architecture

The system is designed as a modular pipeline where each data modality is processed independently and the outputs are combined at the decision level.

```id="q4w8lp"
Skin Image → CNN → Skin Condition (Eczema / Normal)

EDA + Temperature → Machine Learning Model → Stress Level

Motion Data → Machine Learning Model → Scratch Detection

→ Combined Decision → Flare Risk Estimation
```

---

## Datasets

### Skin Image Dataset

A dataset consisting of eczema-affected and normal skin images is used to train the CNN model for classification.

### WESAD Dataset (Referenced)

The WESAD dataset provides electrodermal activity (EDA) and temperature signals. These signals are relevant for modeling physiological stress, which is a known trigger for eczema.

### Human Activity Recognition Dataset (Referenced)

This dataset contains accelerometer and gyroscope data from human activities. It is used conceptually to model scratching behavior based on repetitive motion patterns.

Note: Due to the absence of a unified dataset containing all modalities, the sensor-based models are implemented using simulated data derived from realistic physiological ranges.

---

## Methodology

### Image Classification using CNN

A Convolutional Neural Network is used to learn spatial features from skin images. The architecture includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.

Data augmentation techniques such as rotation, zooming, and horizontal flipping are applied to improve generalization and reduce overfitting.

---

### Stress Detection Model

A machine learning model is used to estimate stress levels based on electrodermal activity and temperature. A Random Forest classifier is employed due to its robustness and ability to handle non-linear relationships.

---

### Scratch Detection Model

Scratching behavior is inferred from motion data using a machine learning classifier. The model identifies patterns associated with repetitive wrist movements.

---

### Decision-Level Fusion

The outputs of the individual models are combined to estimate eczema flare risk. This modular approach avoids direct merging of heterogeneous data and allows each modality to be modeled independently.

---

## Results

The CNN model achieves high classification accuracy, with validation performance indicating good generalization. Data augmentation contributes to improved robustness and reduced overfitting.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy and Pandas
* Matplotlib
* Google Colab

---

## Limitations

* Limited availability of labeled eczema-specific datasets
* Sensor-based models rely on simulated data rather than real-time measurements
* The system is not yet deployed for real-time use

---

## Future Work

* Integration with real-time wearable sensor data
* Deployment as a mobile or web-based application
* Exploration of advanced deep learning architectures such as ResNet
* Incorporation of temporal modeling for continuous monitoring

---

## Conclusion

This project demonstrates a structured approach to eczema monitoring by combining image-based analysis with physiological and behavioral modeling. The modular design provides a foundation for future development of a comprehensive, real-time monitoring system.
