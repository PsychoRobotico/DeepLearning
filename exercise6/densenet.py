from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Activation,
    Convolution2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    concatenate)
from keras.regularizers import l2


def conv_block(x, filters, drop=0, decay=1E-4, name='conv'):
    """ Apply BatchNorm, ReLU, Conv2D and (optionally) Dropout.

    # Arguments
        x: Input Keras tensor
        filters: number of convolution filters
        drop: dropout fraction
        decay: weight decay factor

    # Returns
        Keras tensor after the convolution block.
    """
    x = BatchNormalization(gamma_regularizer=l2(decay),
                           beta_regularizer=l2(decay),
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Convolution2D(filters, (3, 3),
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(decay),
                      name=name + '_conv2D')(x)
    if drop:
        x = Dropout(drop)(x)
    return x


def dense_block(x, num_layers, filters, drop=0, decay=1E-4, name='dense'):
    """ Build a dense_block where each layer is connected to all subsequent layers.

    # Arguments
        x: Keras tensor
        num_layers: the number of convolution_layers in the dense block
        filters: number of extra filters in each subsequent convolution
        drop: dropout fraction
        decay: weight decay factor

    # Returns:
        Keras tensor after the dense block.
    """
    xl = [x]
    for i in range(num_layers):
        _name = '%s_%i' % (name, i + 1)
        x = conv_block(x, filters, drop=drop, decay=decay, name=_name)
        xl.append(x)
        x = concatenate(xl[:], axis=-1, name=_name + '_concat')
    return x


def transition_block(x, filters, drop=None, decay=1E-4, name='transition'):
    """ Apply BatchNorm, ReLU, 1x1 Conv2D, Dropout and Maxpooling2D

    # Arguments
        x: Keras tensor
        filters: number of filters
        drop: dropout fraction
        decay: weight decay factor

    # Returns
        Keras tensor after the transition block
    """
    x = BatchNormalization(gamma_regularizer=l2(decay),
                           beta_regularizer=l2(decay),
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Convolution2D(int(filters), (1, 1),
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(decay),
                      name=name + '_1x1conv')(x)
    if drop:
        x = Dropout(drop)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2), name=name + '_2x2pooling')(x)
    return x


def DenseNet(
        input_shape=(32, 32, 3),
        num_classes=10,
        dense=3,
        layers=12,
        growth=12,
        filters=16,
        compression=1,
        drop=0,
        decay=1E-4):
    """
    Build the DenseNet model

    # Arguments
        input_shape: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        num_classes: number of classes
        dense: number of dense blocks (default 3)
        layers: number of convolution layers per dense block
        growth: number of filters k to add per convolution
        filters: initial number of filters (default 16)
        compression: compression factor of transition blocks (0 - 1)
        drop: dropout fraction
        decay: weight decay

    # Returns
        A Keras model
    """
    x0 = Input(shape=input_shape, name='input')

    # initial convolution
    x = Convolution2D(filters, (3, 3),
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l2(decay),
                      name="initial_conv")(x0)

    for i in range(dense):
        # add dense block
        x = dense_block(x, layers, growth,
                        drop=drop,
                        decay=decay,
                        name='dense_%i' % (i + 1))

        # update the number of filters
        filters += layers * growth
        filters = int(filters * compression)

        # add transition_block, except after last dense_block
        if i < (dense - 1):
            x = transition_block(x, filters,
                                 drop=drop,
                                 decay=decay,
                                 name='transition_%i' % (i + 1))

    x = BatchNormalization(gamma_regularizer=l2(decay),
                           beta_regularizer=l2(decay),
                           name='final_bn')(x)
    x = Activation('relu', name='final_relu')(x)
    x = GlobalAveragePooling2D(name='final_globalpooling')(x)

    # classification layer
    x = Dense(num_classes,
              kernel_regularizer=l2(decay),
              bias_regularizer=l2(decay),
              activation='softmax',
              name='classification')(x)

    return Model(inputs=[x0], outputs=[x], name='DenseNet')


if __name__ == '__main__':
    model = DenseNet()
    model.summary()
