import joblib
from tensorflow.keras.models import load_model

try:
    scaler = joblib.load('scaler.save')
    print('Scaler loaded')
    ae = load_model('lstm_autoencoder.keras')
    print('AE loaded')
    clf = load_model('attack_classifier.keras')
    print('CLF loaded')
    print('AE input shape:', ae.input_shape)
    print('CLF input shape:', clf.input_shape)
    print('CLF output shape:', clf.output_shape)
except Exception as e:
    print('Error:', e)