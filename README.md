# Intrusion Detection System (IDS)

This project implements an Intrusion Detection System using machine learning techniques, specifically an LSTM Autoencoder for anomaly detection and a classifier for attack type identification.

## Features

- Real-time network traffic analysis
- Anomaly detection using LSTM Autoencoder
- Attack classification using trained models
- Web-based interface built with Streamlit

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Karan27q/ids.git
   cd ids
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Web App

To start the Streamlit application:

```
streamlit run app.py
```

This will launch the web interface where you can upload network traffic data for analysis.

### Training Models

The `app.ipynb` notebook contains the code for training the models. Run the cells in order to:

- Load and preprocess data
- Train the LSTM Autoencoder
- Train the attack classifier
- Evaluate and save models

## Files

- `app.py`: Streamlit web application
- `app.ipynb`: Jupyter notebook for model training
- `attack_classifier.keras`: Trained attack classification model
- `lstm_autoencoder.keras`: Trained LSTM autoencoder model
- `scaler.save`: Saved data scaler
- `requirements.txt`: Python dependencies
- `test.py`: Test script

## Dependencies

- streamlit
- tensorflow
- scikit-learn
- joblib
- numpy
- pandas
- matplotlib

## License

[Add license information here]