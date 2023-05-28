import tensorflow as tf
from lib.timing import timing_decorator

@timing_decorator
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