import random
from os import getcwd
import numpy as np
import os
import json
import glob
import shutil
"""
选择方式，修改way的值
"""
way = 1

if way == 1 : # labelme标注的json数据集转为keras yolo的txt训练集  json2txt，obj文件夹下有jpg+json+txt格式文件
    wd = getcwd()
    classes = ["soot"] #修改为待检测的类别名
    image_ids = glob.glob(r"data/obj/*.jpg") #jpg和json文件都在文件夹obj/里
    print(image_ids)
    def convert_annotation(image_id):
        jsonfile=open('%s.json' % (image_id))
        in_file = json.load(jsonfile)
        height=in_file["imageHeight"]
        width=in_file["imageWidth"]
        size=[width,height]
        list_file = open('%s.txt'%(image_id.split('.')[0]), 'w')
        for i in range(0,len(in_file["shapes"])):
            object=in_file["shapes"][i]
            cls=object["label"]
            points=object["points"]
            dw = 1./(size[0])
            dh = 1./(size[1])
            min_x=min_y= np.inf
            max_x = max_y = 0
            for x, y in points:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            x=(min_x+max_x)/2.0
            print(x)
            y=(min_y+max_y)/2.0
            print(y)
            w=max_x-min_x
            h=max_y-min_y
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh
            if cls not in classes:
                print("cls not in classes")
                continue
            cls_id = classes.index(cls)
            b = (x, y, w, h)
            list_file.write(str(cls_id)+" "+" ".join([str(a) for a in b]) )
            list_file.write('\n')
        list_file.close()
        jsonfile.close()

    for image_id in image_ids:
        # list_file.write('%s.jpg' % (image_id.split('.')[0]))
        convert_annotation(image_id.split('.')[0])

if way == 2:# 获得jpg和txt文件，并存入文件夹里，只要txt和jpg
    def moveFileto(sourceDir, targetDir):
        shutil.copy(sourceDir, targetDir)
    objFile = "data/obj/"
    if not os.path.exists(objFile):
        os.makedirs(objFile)
    complete_label_folder = "data/obj"
    list_txt_foder = os.listdir(complete_label_folder)

    for i in range(len(list_txt_foder)):
        if (os.path.splitext(list_txt_foder[i])[1] == ".txt") or (os.path.splitext(list_txt_foder[i])[1] == ".jpg"):
            print(list_txt_foder[i])
            moveFileto(complete_label_folder + "/" + list_txt_foder[i], objFile)

if way == 3:# 从文件夹中将json和jpg文件分别存入jsonFile和imgFiles
    def moveFileto(sourceDir, targetDir):
        shutil.copy(sourceDir, targetDir)

    jsonFile = "data/jsonFile/"
    imgFiles = "data/imgs/"

    if not os.path.exists(jsonFile):
        os.makedirs(jsonFile)
    if not os.path.exists(imgFiles):
        os.makedirs(imgFiles)
    complete_label_folder = "data/obj"
    list_json_foder = os.listdir(complete_label_folder)

    for i in range(len(list_json_foder)):
        if (os.path.splitext(list_json_foder[i])[1] == ".json"):
            print(list_json_foder[i])
            moveFileto(complete_label_folder + "/" + list_json_foder[i], jsonFile)
        elif (os.path.splitext(list_json_foder[i])[1] == ".jpg"):
            print(list_json_foder[i])
            moveFileto(complete_label_folder + "/" + list_json_foder[i], imgFiles)

if way == 4:# 生成保存图片路径的txt文件
    paths = "data/obj/"  # 测试图片的路径,obj包括 jpg和txt文件
    trainFile = open('data/train.txt', 'w')  # 保存图片路径的txt文件
    testFile = open('data/val.txt', 'w')  # 保存图片路径的txt文件

    filenames = os.listdir(paths)

    # 获取txt文件对应的图像文件
    files = []
    for file in filenames:
        if file.split('.')[-1] == 'txt':
            continue
        else:
            files.append(file)
    # 乱序
    random.shuffle(files)

    trainDataList = files[:int(0.8 * len(files))]
    testDataList = files[int(0.8 * len(files)):]

    # 生成训练集
    for filename in trainDataList:
        out_path = "./dataset/v17/obj/" + filename  # 引号内为测试图片文件夹的路径
        print("train:" + out_path)
        trainFile.write(out_path + '\n')
    trainFile.close()

    # 生成测试集
    for filename in testDataList:
        out_path = "./dataset/v17/obj/" + filename  # 引号内为测试图片文件夹的路径
        print("test:" + out_path)
        testFile.write(out_path + '\n')
    testFile.close()


