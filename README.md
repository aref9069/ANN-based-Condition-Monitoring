# ANN-based-Condition-Monitoring
ANN-ConditionMon is an intelligent condition-monitoring and remaining useful life prediction framework for rotating machinery, combining wavelet packet signal processing and artificial neural networks. Designed for vibration-based health monitoring, and predicts future system health and failure horizons.


Remaining Useful Life Estimation of Rolling Element Bearings Wavelet Packet Decomposition + Artificial Neural Network 

This repository contains the implementation of the methodology used in our paper: 

Rohani Bastami A., Aasi A., Arghand H.A. (2018). Estimation of Remaining Useful Life of Rolling Element Bearings
Using Wavelet Packet Decomposition and Artificial Neural Network. Iran Journal of Science and Technology, Electrical Engineering. 
Doi: https://doi.org/10.1007/s40998-018-0108-y

Features

🧩 Artificial Neural Network (ANN): Nonlinear regression for RUL estimation

📈 Vibration Signal Preprocessing: Noise filtering and normalization

🌊 Wavelet Packet Decomposition (WPD): Multiresolution time–frequency analysis

🎯 Kurtosis-Based Optimal Node Selection: Feature extraction and dimensionality reduction

🔁 Reproducible Experiments: Scripts and datasets for benchmark testing



🎯 Key Features

Wavelet Packet Decomposition: Advanced signal processing for feature extraction from vibration signals
Artificial Neural Networks: Deep learning models for RUL prediction
Multi-Feature Analysis:

Root Mean Square (RMS) calculation
Kurtosis factor computation
Statistical moment analysis


Intelligent Smoothing: Data preprocessing with moving window smoothing
Generalized Weibull Fitting: Curve fitting for feature trend modeling
Time-Series Forecasting: Sequential data processing for predictive maintenance
Comprehensive Visualization: Detailed plots for model performance analysis

🛠️ Methodology
1. Signal Processing Pipeline
Raw Vibration Signal → Feature Extraction → Statistical Analysis → Smoothing
2. Feature Extraction

RMS (Root Mean Square): Measures signal energy
Kurtosis: Detects impulsive features indicating bearing defects
Higher-order moments: Captures signal distribution characteristics

3. Neural Network Architecture

Feedforward neural network with optimized hidden layers
Tanh activation for hidden layers
Linear output for regression
Train/Validation/Test split for robust evaluation

4. RUL Prediction

Time-series feature engineering
Sequential pattern learning
Life percentage estimation
Failure horizon prediction



