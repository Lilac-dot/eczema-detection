# Multi-Modal Eczema Monitoring System

## Overview
This project presents a multi-modal system for eczema monitoring and flare risk prediction using image, physiological, and motion data.

## Models
- CNN: Image-based eczema detection
- Random Forest: Stress detection using EDA and temperature
- Random Forest: Scratch detection using motion data

## Methodology
Each model is trained independently on its respective dataset. The outputs are combined using decision-level fusion to estimate flare risk.

## Datasets
- Skin image dataset (eczema vs normal)
- WESAD dataset (EDA, temperature)
- HAR dataset (motion features)

## Output
The system classifies flare risk into:
- Low
- Moderate
- High

## Future Work
- Real-time wearable implementation
- Improved multi-modal datasets
- Advanced deep learning models

## Technologies
- Python
- TensorFlow / Keras
- Scikit-learn
