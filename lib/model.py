from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Activation, Dense, Reshape, Dropout, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


def fire_module(x, fire_id, squeeze=16, expand=64):
    f_id = 'fire' + str(fire_id) + '/'

    # Squeeze
    x = Conv2D(squeeze, (1, 1), padding='valid', name=f_id + 'squeeze1x1')(x)
    x = Activation('relu', name=f_id + 'relu_squeeze1x1')(x)

    # Expand
    left = Conv2D(expand, (1, 1), padding='valid', name=f_id + 'expand1x1')(x)
    left = Activation('relu', name=f_id + 'relu_expand1x1')(left)

    right = Conv2D(expand, (3, 3), padding='same', name=f_id + 'expand3x3')(x)
    right = Activation('relu', name=f_id + 'relu_expand3x3')(right)

    x = Concatenate(axis=-1, name=f_id + 'concat')([left, right])
    return x

def SqueezeNet(input_shape=(224, 224, 3), classes=1000):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    x = Conv2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='softmax')(x)

    model = Model(img_input, x, name='squeezenet')

    return model

def fire_module_next(x, squeeze_channels, expand_channels):
    y = Conv2D(squeeze_channels, 1, activation='relu')(x)
    y1 = Conv2D(expand_channels, 1, activation='relu', padding='same')(y)
    y3 = Conv2D(expand_channels, 3, activation='relu', padding='same')(y)
    return Concatenate()([y1, y3])

def SqueezeNext(input_shape=(32, 32, 3), classes=10):
    input_img = Input(shape=input_shape)
    y = Conv2D(64, 3, strides=2, padding='same')(input_img)
    y = MaxPooling2D(3, strides=2)(y)

    y = fire_module_next(y, 16, 64)
    y = fire_module_next(y, 16, 64)
    y = fire_module_next(y, 32, 128)
    y = MaxPooling2D(3, strides=2)(y)

    y = fire_module_next(y, 32, 128)
    y = fire_module_next(y, 48, 192)
    y = fire_module_next(y, 48, 192)
    y = fire_module_next(y, 64, 256)
    y = MaxPooling2D(3, strides=2)(y)

    y = fire_module_next(y, 64, 256)
    y = GlobalAveragePooling2D()(y)
    y = Reshape((1, 1, 512))(y)
    y = Dropout(0.5)(y)
    y = Conv2D(classes, 1, activation='softmax')(y)
    output = Flatten()(y)

    model = Model(input_img, output)
    return model

def MobileNetV3Small(input_shape=(224, 244, 3), classes=1000):
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, weights='imagenet', include_top=False)

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x) 
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

def evaluate_tflite_model(tflite_model, test_dataset):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    prediction_digits = []
    for test_images, labels in test_dataset.take(1):
        for i, test_image in enumerate(test_images):
            if i % 1000 == 0:
                print(f"Evaluated on {i} results so far.")

            if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                test_image = test_image / input_scale + input_zero_point

            test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
            interpreter.set_tensor(input_details['index'], test_image)

            interpreter.invoke()

            output = interpreter.tensor(output_details['index'])
            digit = np.argmax(output()[0])
            prediction_digits.append(digit == labels[i])

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = prediction_digits.mean()
    return accuracy

def compute_f1_score(tflite_model, test_dataset):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    test_labels = []
    all_predictions = []

    for test_images, labels in test_dataset:
        for i, image in enumerate(test_images):
            if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                image = image / input_scale + input_zero_point

            image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
            interpreter.set_tensor(input_details['index'], image)

            interpreter.invoke()

            predictions = interpreter.get_tensor(output_details['index'])
            predicted_label = np.argmax(predictions)

            test_labels.append(labels[i])
            all_predictions.append(predicted_label)

    f1 = f1_score(test_labels, all_predictions, average='macro')

    return f1
