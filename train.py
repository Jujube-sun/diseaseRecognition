from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import math
from model.GoogleNet import *
from model.MobileNetV2 import *
from model.ShuffleNetV2 import *
from model.SEInception import *
from model.mobilenet_v3_large import *
from keras import optimizers
from keras import losses
import tensorflow as tf
from dataProcess import *
from utils import *
import keras.backend.tensorflow_backend as KTF
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

TRAIN_IMAGES_DIR = './data/train'

TEST_IMAGES_DIR = './data/test'

JSON_TRAIN = './data/train_annotation.json'

JSON_VAL = './data/test_annotation.json'
#归一化尺寸
norm_size=224
#分类数量
class_num=5
# 一次的训练集大小
batch_size = 16
epoch=40
input_shape=(224,224,3)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    #googleNet
    #model=GoogLeNet_build(inputShape=input_shape,classes=class_num)

    #MyNet
    model=SEInception_build(input_shape,class_num)
    
    #mobileNetV2
    #model=MobileNetv2(input_shape=input_shape,k=class_num)

    #mobileNetV1
    #model=MobileNetV1(input_shape=input_shape,classes=class_num)

    #shuffleNetV2
    #model=ShuffleNetV2(input_shape=input_shape,classes=class_num)

    #mobileNetV3_large
    #model=MobileNetV3_Large.build(MobileNetV3_Large(shape=input_shape,n_class=class_num))

    # filename：保存路径保存在机械硬盘
    # monitor：需要监视的值
    # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    # save_best_only：当设置为True时，监测值有改进时才会保存当前的模型
    # period：CheckPoint之间的间隔的epoch数
    #/home/ubuntu,/sata1/yangyongbo/logs/
    checkpoint_period = ModelCheckpoint(
        "/home/ubuntu/logModels" + 'ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5',
        monitor='val_acc',
        save_weights_only=True,
        mode='auto',
        save_best_only=True,
        period=5)

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    # monitor：需要监视的值
    # factor：学习率每次降低多少，new_lr = old_lr * factor
    # patience：容忍网路的性能不提升的次数，高于这个次数就降低学习率
    # verbose（bool）：如果为1，则为每次更新向stdout输出一条消息。 默认值：0
    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc',
        mode='auto',
        factor=0.5,
        patience=3,
        verbose=1)

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    # monitor: 需要监视的量，val_loss，val_acc
    # min_delta：增大或减小的阈值，只有大于这个部分才算作提升
    # patience: 当early stop被激活(如发现loss/val相比上一个epoch训练没有下降/上升,)，则经过 patience 个 epoch 后停止训练
    # verbose（bool）：如果为1，则为每次更新向stdout输出一条消息。 默认值：0
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=4,
        verbose=1)

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='logs/tensorboard')
    callback_lists=[tensorboard,reduce_lr,early_stopping]

    #constructed dataset
    train_image_path, train_labels = process_annotation('./data/train_annotation_NoLeaf.json',
        './data/train',class_num)
    test_image_path, test_labels = process_annotation('./data/test_annotation_NoLeaf.json',
        './data/test', class_num)
    #plantvillage
    # train_image_path, train_labels = process_annotation_plant('./data/plantVillageData/train_annotation_plant.json',
    #     './data/plantVillageData/plantVillage',class_num)
    # test_image_path, test_labels = process_annotation_plant('./data/plantVillageData/valid_annotation_plant.json',
    #     '.data/plantVillageData/plantVillage', class_num)
    model.fit_generator(generator=generate_load_date(train_image_path, train_labels,norm_size,batch_size),
                        steps_per_epoch=len(train_labels)//batch_size,epochs=epoch,
                        validation_data=generate_load_date(test_image_path, test_labels,norm_size,batch_size),
                        validation_steps=len(test_labels)//batch_size,callbacks=[tensorboard,reduce_lr,early_stopping])

    model.save(log_dir + 'Base_Plant.h5')
