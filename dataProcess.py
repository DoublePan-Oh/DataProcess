from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np
import time
import random
import shutil

imageDir="G:\\posture_detection\\res_phone\\20210315\\phone_coco_1\\" #要改变的图片的路径文件夹
saveDir="G:\\posture_detection\\res_phone\\20210315\\phone_coco_augmentation"   #要保存的图片的路径文件夹

i=0

def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def move(root_path,img_name,off): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = img.offset(off,0)
    return offset

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(60) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

def contrastEnhancement(root_path,img_name):#对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def colorEnhancement(root_path,img_name):#颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored

way = 1

if way == 1:
    """
    数据增强
    """
    for name in os.listdir(imageDir):
        i=i+1
        saveName="flip_"+name
        saveImage=flip(imageDir,name) #翻转
        print(saveName)
        saveImage = saveImage.convert('RGB')
        saveImage.save(os.path.join(saveDir,saveName))

if way == 2:
    """
    resize 图像
    """
    for name in os.listdir(imageDir):
        i = i + 1
        saveName = "resize_" + name
        image = cv2.imread(os.path.join(imageDir, name))
        saveImage = cv2.resize(image, (128, 128))
        print(saveName)
        cv2.imwrite(saveDir + saveName, saveImage)

if way == 3:
    """
    从视频流抽帧
    """
    times = 0
    frameFrequency = 25 # 每隔25帧抽一次
    video_path = "G:\posture_detection\Security_Detector\ch06_20210306134701.mp4"
    outPutDirName = 'G:\posture_detection\Security_Detector\img\\'

    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)# 如果文件目录不存在则创建目录

    camera = cv2.VideoCapture(video_path)
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            localtime = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            cv2.imwrite(outPutDirName + str(localtime) + str(times) + '.jpg', image)
            print(outPutDirName + str(localtime) + '.jpg')
    print('图片提取结束')
    camera.release()

if way == 4:
    """
    划分训练集和验证集（测试集）,并将图片复制到其他路径下
    """
    paths = "G:\\posture_detection\\res_phone\\20210311\\jiangcun\\"  # 测试图片的路径
    filenames = os.listdir(paths)

    # 获取txt文件对应的图像文件
    files = []
    for file in filenames:
        files.append(file)
    random.shuffle(files)# 乱序

    trainDataList = files[:int(0.8 * len(files))]
    testDataList = files[int(0.8 * len(files)):] # 8:2划分训练集和测试集

    for i in trainDataList:
        aa = i.split(".")[0]
        oldname = paths + aa + ".jpg"
        newname = "G:\\posture_detection\\res_phone\\20210311\\train\\nophone\\" + aa + ".jpg"
        shutil.copyfile(oldname, newname)
        print("train:" + newname)

    for j in testDataList:
        aa = j.split(".")[0]
        oldname = paths + aa + ".jpg"
        newname = "G:\\posture_detection\\res_phone\\20210311\\test\\nophone\\" + aa + ".jpg"
        shutil.copyfile(oldname, newname)
        print("test:" + newname)

