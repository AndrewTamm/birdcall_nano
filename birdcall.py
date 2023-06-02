from lib import load, model, plot, quantize, train
import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
import numpy as np


if __name__ == "__main__":

    #call_recordings = fetch_data()

    #download_files(call_recordings, 'calls')


    X_train, X_test, y_train, y_test, label_encoder = load.multi_load('calls', pool_size=10)

    plot.plot(X_train, X_test, y_train, y_test, label_encoder)

    # One-hot encode the labels
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    # Add an extra dimension to the spectrogram data
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Add an extra dimension to the spectrogram data
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    
    
    sq = model.SqueezeNet(input_shape=X_train[0].shape, classes=len(label_encoder.classes_))
    #sq.summary()

    train.train(sq, X_train, X_test, y_train, y_test)

    qm = quantize.quantize_model(sq, "test.tflite", X_train)
    print(quantize.get_tflite_model_size(tf.lite.TFLiteConverter.from_keras_model(sq).convert()))
    print(quantize.get_tflite_model_size(qm))
    print(len(label_encoder.classes_))