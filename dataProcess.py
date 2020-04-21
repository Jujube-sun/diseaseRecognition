import json
# encoding: utf-8

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical#相当于one-hot
import cv2
import numpy as np
import random
import os

#获取图片的路径和标签，用于分批次训练，载入内存
#[./data/test/Healthy/bea07a8790160e19f09cd08a10d27ac3.jpg],[5]
def process_annotation(anno_file, dataset_dir,class_num):
    #需要传入标注文件所在路径，以及训练/测试图片所在路径
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            #path=os.path.join(dataset_dir,anno["disease_name"])
            path = dataset_dir+"/"+anno["disease_name"]
            img_paths.append(path + "/" +anno["image_id"])
            labels.append(anno["disease_class"])
        labels = np.array(labels)
        labels = to_categorical(labels, num_classes=class_num)  # one-hot

    return img_paths, labels


#获取图片的路径和标签，用于分批次训练，载入内存
#[./data/test/Healthy/bea07a8790160e19f09cd08a10d27ac3.jpg],[5]
def process_annotation_plant(anno_file, dataset_dir,class_num):
    #需要传入标注文件所在路径，以及训练/测试图片所在路径
    with open(anno_file) as file:
        annotations = json.load(file)
        #打乱顺序
        random.shuffle(annotations)
        img_paths = []
        labels = []
        for anno in annotations:
            #path=os.path.join(dataset_dir,anno["disease_name"])
            path = dataset_dir+"/"+anno["disease_class"]+"_"+anno["disease_name"]
            img_paths.append(path+"/"+anno["image_id"])
            #print(img_paths)
            labels.append(anno["disease_class"])
        labels = np.array(labels)
        labels = to_categorical(labels, num_classes=class_num)  # one-hot

    return img_paths, labels

#一次性把数据载入内存
def load_data(anno_file, dataset_dir,norm_size,class_num):
    with open(anno_file) as file:
        annotations = json.load(file)
        data = []#数据
        labels = []
        for anno in annotations:
            path = dataset_dir + "/" + anno["disease_name"]+ "/" + anno["image_id"]
            image = cv2.imread(path)  # 读取文件
            image = cv2.resize(image, (norm_size, norm_size))  # 统一图片尺寸
            image = img_to_array(image)
            data.append(image)
            labels.append(anno["disease_class"])
        data = np.array(data, dtype="float") / 255.0  # 归一化
        labels = np.array(labels)
        labels = to_categorical(labels,num_classes=class_num)  # one-hot
    return data, labels

#分批次载入内存
def generate_load_date(img_paths, labels,norm_size,batch_size):
    '''
        参数：
            image_paths：所有图片路径列表
            labels: 所有图片对应的标签列表
            batch_size:批次
            norm_size:图片归一化尺寸
        返回:
            一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
        '''
    while 1:
        for i in range(0,len(img_paths),batch_size):
            batch_data=get_im_cv2(img_paths[i:i+batch_size],norm_size)
            batch_labels=labels[i:i+batch_size]
            yield (np.array(batch_data)/255.0, batch_labels)


#基于keras的数据扩充
def augmentation_data(load_path,save_path):
    #图片生成器
    #horizontal_flip，随机水平翻转;vertical_flip,随机数值旋转
    #rotation_range=40，不超过40度的旋转
    #zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，
    #width_shift_range水平和竖直偏移幅度
    # 若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    datagen = ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        width_shift_range=0.2,
        height_shift_range=0.2)


    gen = datagen.flow_from_directory(load_path,target_size = (224, 224),batch_size = 32,
                                      save_to_dir = save_path,save_prefix = 'gen',save_format = 'jpg')
    for i in range(18):
        gen.next()
        i+1


# 读取图片函数
def get_im_cv2(paths,norm_size):
    '''
    参数：
        paths：要读取的图片路径列表
        norm_size:图片归一化尺寸
    返回:
        imgs: 图片数组
    '''
    # Load as grayscale
    imgs = []
    for path in paths:

        image = cv2.imread(path)
        try:
            # Reduce size
            image = cv2.resize(image, (norm_size, norm_size))  # 统一图片尺寸
            imgs.append(image)
        except cv2.error:
            print(path)
            print(str(cv2.error))

    return imgs