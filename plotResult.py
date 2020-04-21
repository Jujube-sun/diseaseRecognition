import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x, y


if __name__=="__main__":
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    x1, y1 = readcsv("./result/loss/inModels/Base.csv")
    #x1, y1 = readcsv("./result/loss/myData/MobileNetV1.csv")
    #x1, y1 = readcsv("./result/loss/plantVillage/MobileNetV1.csv")
    plt.plot(x1, y1, color="black", label='Base')

    x2, y2 = readcsv("./result/loss/inModels/Base_Multi.csv")
    #x2, y2 = readcsv("./result/loss/myData/MobileNetV2.csv")
    #x2, y2 = readcsv("./result/loss/plantVillage/MobileNetV2.csv")
    plt.plot(x2, y2, color='blue', label='Base_Multi')

    x3, y3 = readcsv("./result/loss/inModels/Base_Multi_BN.csv")
    # x3, y3 = readcsv("./result/loss/myData/ShuffleNetV2.csv")
    # x3, y3 = readcsv("./result/loss/plantVillage/ShuffleNetV2.csv")
    plt.plot(x3, y3, color='green', label='Base_Multi_BN')

    x4, y4 = readcsv("./result/loss/inModels/MyNet.csv")
    #x4, y4 = readcsv("./result/loss/myData/MyNet.csv")
    #x4, y4 = readcsv("./result/loss/plantVillage/MyNet.csv")
    plt.plot(x4, y4, color='red', label='MyNet')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.ylim(0, 1.8)
    plt.xlim(0, 30)
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=16)
    plt.show()