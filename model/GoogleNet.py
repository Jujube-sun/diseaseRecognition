from keras.layers import Dropout, BatchNormalization, concatenate, AveragePooling2D
from keras.layers import Dense, Conv2D, MaxPooling2D,Input,Flatten
from keras.models import Model

def GoogLeNet_build(inputShape, classes):
    '''
    GoogLeNet网络
    INPUT  -> 输入数据格式(224, 224, 3), 待分类数(1000)
    '''
    #定义inception模块，已采取的措施在inceptionv1基础上增加了归一化，
    #没有采取的措施是在论文中3*3，5*5的之前的1*1以及maxpooling后面的1*1现已加上
    def Inception_block(inputs, num_filter):
        branch1x1 = Conv2D(num_filter, kernel_size=(1, 1), activation='relu', strides=(1, 1), padding='same')(inputs)
        #branch1x1 = BatchNormalization(axis=3)(branch1x1)

        #先进行1*1然后进行3*3
        branch3x3 = Conv2D(num_filter, kernel_size=(1, 1), activation='relu', strides=(1, 1), padding='same')(inputs)
        branch3x3 = Conv2D(num_filter, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(branch3x3)
        #branch3x3 = BatchNormalization(axis=3)(branch3x3)

        branch5x5 = Conv2D(num_filter, kernel_size=(1, 1), activation='relu', strides=(1, 1), padding='same')(inputs)
        branch5x5 = Conv2D(num_filter, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(branch5x5)
        #branch5x5 = BatchNormalization(axis=3)(branch5x5)

        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1, 1), padding='same')(inputs)
        branchpool = Conv2D(num_filter, kernel_size=(1, 1), activation='relu', strides=(1, 1), padding='same')(branchpool)

        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
        return x

    inputs = Input(shape=inputShape)

    x = Conv2D(64, kernel_size=(7, 7), activation='relu', strides=(2, 2), padding='same')(inputs)
    #x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(192, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    #x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception_block(x, 64)
    x = Inception_block(x, 120)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    x = Inception_block(x, 128)
    x = Inception_block(x, 128)
    x = Inception_block(x, 128)
    x = Inception_block(x, 132)
    x = Inception_block(x, 208)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    x = Inception_block(x, 208)
    x = Inception_block(x, 256)

    x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
    x = Dropout(0.4)(x)
    #分类器
    x = Flatten()(x)  # 特征扁平化
    x = Dense(classes,activation='softmax')(x)  # 全连接层，进行多分类,形成最终的10分类

    model = Model(inputs, x)
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model


# if __name__=="__main__":
#     GoogLeNet_build(inputShape=(224,224,3),classes=6)