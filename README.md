# Neuromorphic Crop Recommendation with Spiking Neural Networks

A machine-learning research project that explores crop recommendation using a **Spiking Neural Network (SNN)** and compares it with conventional machine-learning models.

## Overview

The system uses agricultural features such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall to recommend a crop. The project evaluates a neuromorphic learning approach alongside standard classifiers.

## Input Features

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

## Models

The training pipeline includes:

- Spiking Neural Network using **snnTorch**
- Random Forest
- Support Vector Machine
- Multi-Layer Perceptron

The SNN converts normalized feature values into spike trains and processes them through multiple leaky integrate-and-fire layers.

## Technology Stack

- Python
- PyTorch
- snnTorch
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Flask

## Project Structure

```text
Neuromorphic_SNN/
├── train.py
├── app.py
├── model_utils.py
├── dataset/
├── crop_snn_best.pth
├── rf_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── model_config.pkl
├── training_curves.png
├── confusion_matrix.png
├── model_comparison.png
└── f1_precision_per_class.png
```

## Training

```bash
python train.py
```

The training script:

1. Loads and preprocesses the crop dataset.
2. Standardizes numerical features.
3. Trains conventional baseline models.
4. Rate-encodes features as spike trains.
5. Trains the SNN.
6. Evaluates performance and saves model artifacts and plots.

## Why Neuromorphic Computing?

Spiking neural networks process information through discrete spike events inspired by biological neurons. This project investigates how that approach can be applied to a practical agricultural classification problem.

## Future Improvements

- Benchmark energy efficiency on neuromorphic hardware
- Add more agricultural and soil variables
- Perform broader cross-validation and hyperparameter studies
- Deploy the inference pipeline to edge hardware
- Expand model interpretability and error analysis

## Author

**Sadik Shaik**

Computer Engineering / AI & Embedded Systems Projects
