from lib import load, model, plot, quantize, train, pruning, util
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoder = load.multi_load('calls', pool_size=10)

    #plot.plot(X_train, X_test, y_train, y_test, label_encoder)

    # Add an extra dimension to the spectrogram data
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Add an extra dimension to the spectrogram data
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    sq = model.SqueezeNet(input_shape=X_train[0].shape, classes=len(label_encoder.classes_))
    #sq.summary()

    train.train(sq, X_train, X_test, y_train, y_test, 32, 100)
    print(f"original: {util.get_tflite_model_size(tf.lite.TFLiteConverter.from_keras_model(sq).convert())}")
    print(f"original zipped: {util.get_gzipped_model_size(tf.lite.TFLiteConverter.from_keras_model(sq).convert())}")

    original_accuracy = model.evaluate_tflite_model(tf.lite.TFLiteConverter.from_keras_model(sq).convert(), X_test, y_test)
    print(f"accuracy before pruning: {original_accuracy}")

    ######## Quantize Only ########
    qm = quantize.quantize_model(sq, "test.tflite", X_train)
    print(f"quantized: {util.get_gzipped_model_size(qm)}")
    util.save_tflite_model(qm, "quantized.tflite")
    
    accuracy_after_quantization = model.evaluate_tflite_model(qm, X_test, y_test)
    print(f"accuracy after quantization: {accuracy_after_quantization}")

    ''' skip since tflite can't read a zip file
    ######## Prune and then Quantize ########
    pruned = pruning.apply_pruning(sq, X_train, y_train, X_test, y_test, 32, 100)
    print(f"pruned: {util.get_gzipped_model_size(tf.lite.TFLiteConverter.from_keras_model(pruned).convert())}")

    qm = quantize.quantize_model(pruned, "test.tflite", X_train)
    print(f"quantized: {util.get_gzipped_model_size(qm)}")
    util.save_tflite_model(qm, "quantized.tflite")
    
    accuracy_after_quantization = model.evaluate_tflite_model(qm, X_test, y_test)
    print(f"accuracy after quantization: {accuracy_after_quantization}")
    '''
    
    print(f"number of classes: {len(label_encoder.classes_)}")
