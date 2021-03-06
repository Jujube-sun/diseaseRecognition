from keras.models import Model
from keras import backend as K
from utils import *
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten,Conv2D, MaxPooling2D, AveragePooling2D, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3
import numpy as np

# seed = 7
# np.random.seed(seed)
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    """卷积+BN层

        # Arguments
            x: input tensor.输入
            nb_filters: filters in `Conv2D`.卷积核的数量
            kernel_size: height and width of the convolution kernel.卷积核大小
            padding: padding mode in `Conv2D`.填充形式默认为same
            strides: strides in `Conv2D`.步长
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.

        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    #base
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x


def Multi_Shape(x, big_nb_filter, small_nb_filter, channel_axis):
    branch1x1 = Conv2d_BN(x, small_nb_filter, (1, 1), padding="same", strides=(2, 2), name=None)

    branch3x3 = Conv2d_BN(x, small_nb_filter, (3, 3), padding="same", strides=(2, 2), name=None)

    branch5x5 = Conv2d_BN(x, big_nb_filter, (5, 5), padding="same", strides=(2, 2), name=None)

    branch7x7 = Conv2d_BN(x, big_nb_filter, (7, 7), padding="same", strides=(2, 2), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branch7x7], axis=channel_axis)
    return x


# 最基础的Inception
def Inception_Base(x, nb_filter, channel_axis):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (5, 5), padding='same', strides=(1, 1), name=None)

    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=channel_axis)

    return x


# 5*5卷积变为了两个3*3卷积
def Inception_A(x, nb_filter, channel_axis):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=channel_axis)

    return x


# 5*5卷积变为了两个3*3卷积且strdie=2
def Inception_A2(x, nb_filter, channel_axis):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(2, 2), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(2, 2), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (3, 3), padding='same', strides=(2, 2), name=None)

    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=channel_axis)

    return x


def SEInception_build(inputShape,classes):
    inputs = Input(inputShape)
    # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x=Multi_Shape(inputs,32,16,channel_axis)
    #x = Conv2d_BN(inputs,64, kernel_size=(7, 7), strides=(2, 2))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception_A(x, 64,channel_axis)  # 256
    x =senet(16,x)

    x = Inception_A2(x, 96,channel_axis)  # 480
    x = senet(16, x)

    x = Inception_A2(x, 120,channel_axis)  # 512
    x = senet(16, x)
    # x = Inception_A2(x, 128,channel_axis)  # 528
    #x = senet(16, x)

    #x = senet(16, x)
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Dropout(0.4)(x)
    #拉成一维之前特征图大小最好为1x1，减少最后一层的全连接的参数数量
    x = Flatten()(x)
    #x = Dense(classes, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x, name='newInception')
    # model.compile(loss=Focal_Loss,
    #               optimizer='sgd',
    #               metrics=['accuracy'])
    model.summary()

    return model