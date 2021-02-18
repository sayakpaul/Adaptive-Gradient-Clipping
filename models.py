from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def conv_block(x, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
               kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    return x

def conv_block_bn(x, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_mini_vgg(use_bn=True):
    inputs = Input((96, 96, 3))

    for _ in range(2):
        if use_bn:
            x = conv_block_bn(inputs, 32)
        else:
            x = conv_block(inputs, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(2):
        if use_bn:
            x = conv_block_bn(inputs, 64)
        else:
            x = conv_block(inputs, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(5, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model