# ANN-based-Condition-Monitoring
ANN-ConditionMon is an intelligent condition-monitoring and remaining useful life prediction framework for rotating machinery, combining wavelet packet signal processing and artificial neural networks. Designed for vibration-based health monitoring, and predicts future system health and failure horizons.


Remaining Useful Life Estimation of Rolling Element Bearings Wavelet Packet Decomposition + Artificial Neural Network 

This repository contains the implementation of the methodology used in our paper: 
Rohani Bastami A., Aasi A., Arghand H.A. (2018). Estimation of Remaining Useful Life of Rolling Element Bearings
Using Wavelet Packet Decomposition and Artificial Neural Network. Iran Journal of Science and Technology, Electrical Engineering.

Features

🧩 Artificial Neural Network (ANN): Nonlinear regression for RUL estimation

📈 Vibration Signal Preprocessing: Noise filtering and normalization

🌊 Wavelet Packet Decomposition (WPD): Multiresolution time–frequency analysis

🎯 Kurtosis-Based Optimal Node Selection: Feature extraction and dimensionality reduction

🔁 Reproducible Experiments: Scripts and datasets for benchmark testing


📁 Repository Structure

ANN-ConditionMon/
│
├── data/                     # Vibration datasets (raw and processed)
├── preprocessing/             # Signal denoising and normalization scripts
├── wpt_features/              # Wavelet packet decomposition and feature extraction
├── models/                    # ANN architecture and training code
├── results/                   # Experimental outputs and plots
├── utils/                     # Helper functions (e.g., metrics, plotting)
├── main.py                    # Entry point for the full pipeline
└── README.md                  # Project documentation



