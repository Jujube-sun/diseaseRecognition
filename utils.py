import  cv2
import os
import tensorflow as tf
import json
import random
from tensorflow.python.ops import array_ops
from keras import backend as K
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import Model
from keras.layers import Layer,Dense,GlobalAveragePooling2D,Multiply,Conv2D,\
    Reshape,GlobalMaxPooling2D,Add,Activation,Input
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
from pylab import *
import datetime
import time
from dataProcess import *
import math
from sklearn.metrics import classification_report



# #GPU:2080需要，1080ti不用
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.tensorflow_backend.set_session(tf.Session(config=config))

def rename():
    #注意路径用这个/
    #重命名图片文件
    rootdir = './data/plantVillage'
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        num = 0
        path = os.path.join(rootdir, list[i])
        disease_class, disease_name = list[i].split("_", 1)
        #print(path)
        # 判断是否是文件如果是文件则继续打开该文件
        if not os.path.isfile(path):
            # 获得某类病害下的所有图片路径
            littleList = os.listdir(path)
            for j in range(0, len(littleList)):
                image_id = littleList[j]
                newName=disease_name+str(num)+".JPG"
                os.rename(os.path.join(path,image_id),os.path.join(path,newName))
                num+=1

def read_image(image_path,norm_size):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (norm_size, norm_size))  # 统一图片尺寸
    image = image.astype("float") / 255.0  # 归一化
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


def getDiseaseName(disease_class):
    if disease_class==0:
        disease_name="ChloroticVirus"
    elif disease_class==1:
        disease_name="CottonBlight"
    elif disease_class==2:
        disease_name="LeafMiner"
    elif disease_class==3:
        disease_name="PowderyMildew"
    elif disease_class==4:
        disease_name="Whitefly"
    elif disease_class==5:
        disease_name="Healthy"
    return disease_name


#se模块keras实现
def senet(reduction,input):
    channels = input.shape.as_list()[-1]
    #avg_x输出的是(None,256)
    avg_x = GlobalAveragePooling2D()(input)
    #为了得到(None,1,1,256)这个值
    avg_x = Reshape((1,1,channels))(avg_x)
    #用1*1的卷积层代替全连接层(None,1,1,16)
    avg_x = Conv2D(filters=int(channels)//reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(avg_x)
    #(None,1,1,256)
    avg_x = Conv2D(filters=int(channels),kernel_size=(1,1),strides=(1,1),padding='valid')(avg_x)

    cbam_feature = Activation('sigmoid')(avg_x)

    return Multiply()([input,cbam_feature])

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


#创建每一个字典
def setInfo(image_id,disease_class,disease_name):
    demo = {}
    demo["image_id"]=image_id
    demo["disease_class"]=disease_class
    demo["disease_name"]=disease_name
    return demo


def setDiseaseClass2(disease_name):
    if disease_name=="YellowSmut":
        disease_class=0
    elif disease_name=="CottonBlight":
        disease_class=1
    elif disease_name=="PowderyMildew":
        disease_class=2
    elif disease_name=="Whitefly":
        disease_class=3
    elif disease_name=="Healthy":
        disease_class=4
    return disease_class


def setMyJson(rootdir,savePath,filename):
    annotation = []
    # 保存的json名称及路径
    filename = savePath + "/" + filename
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        # print(list[i])
        #以下两句为设置自己json数据格式
        disease_name = list[i]
        disease_class = setDiseaseClass2(disease_name)

        path = os.path.join(rootdir, list[i])
        # 判断是否是文件如果是图片则继续操作
        if not os.path.isfile(path):
            # 获得某类病害下的所有图片路径
            littleList = os.listdir(path)
            # 打乱路径
            random.shuffle(littleList)
            for j in range(0, len(littleList)):
                image_id = littleList[j]
                annotation.append(setInfo(image_id=image_id, disease_class=disease_class, disease_name=disease_name))

    # 随机打乱train和test的保存顺序
    random.shuffle(annotation)
    #print(len(annotation))
    # print(len(annotation))
    with open(filename, 'w') as file_obj:
        json.dump(annotation, file_obj)

#生成plantvillage的json
def setPlantJson(rootdir,savePath):
    trainannotation = []
    validannotation = []
    testannotation = []
    # 保存的json名称及路径
    trainfilename = savePath + "/" + "train_annotation_plant.json"
    validfilename = savePath + "/" + "valid_annotation_plant.json"
    testfilename  = savePath + "/" + "test_annotation_plant.json"
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        # print(list[i])
        #0_Apple___Apple_scab分为0和Apple___Apple_scab，以后如果分三级可以再细分
        disease_class, disease_name= list[i].split("_",1)
        #print(disease_name+"&"+disease_class)

        path = os.path.join(rootdir, list[i])
        # 判断是否是文件如果是文件则继续打开该文件
        if  not os.path.isfile(path):
            # 获得某类病害下的所有图片路径
            littleList = os.listdir(path)
            # 打乱一个类内的路径
            random.shuffle(littleList)
            trainNumber=int(len(littleList)*0.6)
            validNumber=int(len(littleList)*0.2)
            #print(str(trainNumber)+"&&"+str(validNumber))
            for j in range(0, trainNumber):
                image_id = littleList[j]
                #print(image_id)
                trainannotation.append(setInfo(image_id=image_id, disease_class=disease_class, disease_name=disease_name))
            for k in range(trainNumber, trainNumber+validNumber):
                image_id = littleList[k]
                validannotation.append(setInfo(image_id=image_id, disease_class=disease_class, disease_name=disease_name))
            for l in range(trainNumber+validNumber, len(littleList)):
                image_id = littleList[l]
                testannotation.append(setInfo(image_id=image_id, disease_class=disease_class, disease_name=disease_name))
    # 随机打乱train,valid,test的保存顺序
    random.shuffle(trainannotation)
    random.shuffle(validannotation)
    random.shuffle(testannotation)
    #print(len(trainannotation))
    with open(trainfilename, 'w') as file_obj:
        json.dump(trainannotation, file_obj)
    with open(validfilename, 'w') as file_obj:
        json.dump(validannotation, file_obj)
    with open(testfilename, 'w') as file_obj:
        json.dump(testannotation, file_obj)


def predictSinglePhoto(label_file,image_id,image_path,model_path):
    norm_size=224
    disease_class=0
    disease_name=" "
    image=read_image(image_path,norm_size)
    output = cv2.resize(image, (600, 600))

    # 读取模型和标签
    print("------读取模型和标签------")
    model = load_model(model_path)
    #mobileNet.定义relu6
    #model=load_model(model_path,custom_objects={'relu6': relu6})

    with open(label_file) as file:
        annotations = json.load(file)
        for anno in annotations:
            if anno["image_id"] == image_id:
                disease_name=anno["disease_name"]
                disease_class=anno["disease_class"]
                break


    # 预测
    #start_time = datetime.datetime.now()
    t1 = datetime.datetime.now().microsecond
    t3 = time.mktime(datetime.datetime.now().timetuple())
    preds = model.predict(image)
    t2 = datetime.datetime.now().microsecond
    t4 = time.mktime(datetime.datetime.now().timetuple())
    strTime = 'funtion time use:%dms' % ((t4 - t3) * 1000 + (t2 - t1) / 1000)
    print(strTime)

    # 得到预测结果以及其对应的标签
    i = preds.argmax(axis=1)[0]
    if i == disease_class:
        message = disease_name
    else:
        message="fasle:predict disease is "+str(i)


    # 在图像中把结果画出来
    text = "{}: {:.2f}%".format(message, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 1)

    # 绘图
    cv2.imshow("Image", output)
    cv2.waitKey(0)


def preClassAcc(model_path,JSON_VAL,TEST_IMAGES_DIR,norsize,class_num):
    model=load_model(model_path)
    #mobileNet.定义relu6
    #model=load_model(model_path,custom_objects={'relu6': relu6})
    # Batch size (you should tune it based on your memory)
    batch_size = 16

    #分批次读入内存做预测
    image_paths, labels = process_annotation(JSON_VAL, TEST_IMAGES_DIR, class_num)
    # #steps：//表示向下取整,ceil表示向上取整
    steps=math.ceil(len(labels)/batch_size)
    validation_generator=generate_load_date(image_paths, labels, norsize, batch_size)
    predictions = model.predict_generator(generator=validation_generator, steps=steps)
    # 把预测出来的，最高一列的那个数值变为1其他的数值变为0
    for i in range(len(predictions)):
        max_value = max(predictions[i])
        for j in range(len(predictions[i])):
            if max_value == predictions[i][j]:
                predictions[i][j] = 1
            else:
                predictions[i][j] = 0
    target_names = ['YellowSmut', 'CottonBlight', 'PowderyMildew', 'Whitefly', 'Healthy']
    result = classification_report(y_true=labels, y_pred=predictions,target_names=target_names,digits=4)
    return result


# if __name__ =='__main__':
#     label_file="./data/test_annotation_NoLeaf.json"
#     image_id="CUW36.JPG"
#     image_path="./data/test/PowderyMildew/"+image_id
#     model_path="./logs/ShuffleNetV1_Mine.h5"
#     predictSinglePhoto(label_file=label_file,image_id=image_id,
#                        image_path=image_path,model_path=model_path)


