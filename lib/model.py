from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Activation, Dense
from tensorflow.keras.models import Model

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
