import tensorflow as tf
from pathlib import Path


def quantize_model(model, save_path, X_train):

    def representative_dataset_gen():
            for i in range(100):
                yield [X_train[i:i + 1]]

    # Convert the SavedModel to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Save the quantized model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model

def get_tflite_model_size(tflite_model):
    tflite_model_file = Path("temp.tflite")
    with tflite_model_file.open("wb") as f:
        f.write(tflite_model)
    size = tflite_model_file.stat().st_size
    tflite_model_file.unlink()
    return size