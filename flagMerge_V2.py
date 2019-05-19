import os
import cv2
import numpy as np 
import csv
import random


from imgaug import augmenters as iaa


def main():
    # 更改旗子名字
    country = 'zhongguo'

    video_path = './video/' + country + '.avi'    # 视频路径
    background_path = 'I:\\coco\\test2017\\' # 背景路径

    # 定义保存路径
    saveBoundRect_path = 'I:/flag/boundRect/'    # 边框矩形保存路径
    saveMinRect_path = 'I:/flag/minRect/'# 最小外接矩形保存路径
    saveMergePic_path = 'I:/flag/mergePic/'# 合成图像保存路径
    saveCSV_path = 'I:/flag/csv/' #csv保存路径
 
    # 创建路径
    if not os.path.exists(saveMinRect_path):
        os.makedirs(saveMinRect_path)
    if not os.path.exists(saveBoundRect_path):
        os.makedirs(saveBoundRect_path)
    if not os.path.exists(saveMergePic_path):
        os.makedirs(saveMergePic_path)
    if not os.path.exists(saveCSV_path):
        os.makedirs(saveCSV_path)

    # 读取背景路径下文件列表
    background_list = os.listdir(background_path)
    
    # 打开csv文件
    csvfile = open(saveCSV_path + country + '.csv', 'w', newline='')  
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['path','x1', 'y1', 'x2', 'y2', 'label'])   #先写入columns_name

    
    frame_cnt = 0   # 帧数
    for i in range(50): # 循环一次生成101张
        #打开视频
        cap = cv2.VideoCapture(video_path)   
        while(True):
            # 捕获一帧图像
            ret, frame = cap.read()  

            if not ret: # 捕获失败退出
                break

            frame_cnt += 1 # 帧数
            # 拷贝原始帧副本
            src_image = frame.copy()
            # 预处理生成二值化图像
            binary_image = toBinary(src_image, threshold=10)
            # 寻找最大轮廓
            maxContour = findMaxContour(binary_image)
            cv2.drawContours(src_image, [maxContour], 0, (255,0,0), 2)  
            # cv2.imshow('maxContour', src_image)
            
            # 获得轮廓的外接矩形
            boundRect = cv2.boundingRect(maxContour) # boundRect = (x, y, w, h)
            minRect = cv2.minAreaRect(maxContour)    # minRect[0] = (x, y), minRect[1] = (w, h), minRect[2] = angle([-90,0])
            # 绘制外接矩形 
            # drawRect(src_image, boundRect, minRect)

            # 裁剪并保存最小外接矩形
            minRect_image = cutOutMinRect(frame, minRect)
            cv2.imwrite(saveMinRect_path + country + '_minRect_' + str(frame_cnt) + '.jpg', minRect_image)
            # 裁剪并保存边框矩形截图
            boundRect_image = cutOutBoundRect(frame, boundRect)
            cv2.imwrite(saveBoundRect_path + country + '_boundRect_' + str(frame_cnt) + '.jpg', boundRect_image)



            # 随机选取背景图片
            background_image = selectBackground(background_path, background_list, boundRect_image)

            wb = background_image.shape[1]
            hb = background_image.shape[0]

            ratiow = int(wb/boundRect[2])
            ratioh = int(hb/boundRect[3])

            # 填充旗子轮廓作为掩膜

            boundRect_image_mask = fillContourToMask(frame, maxContour, boundRect)

            # 图像缩放和增广
            boundRect_image, boundRect_image_mask = imageAugment(boundRect_image, boundRect_image_mask, ratiow, ratioh, 0)

            # 合成图像
            merge_image, flagPosition = mergeImage(boundRect_image, boundRect_image_mask, background_image)
            cv2.imwrite(saveMergePic_path + country + '_merge_' + str(frame_cnt) + '.jpg', merge_image)

            # 写入csv
            image_path = './mergePic/' + country + '_merge_' + str(frame_cnt) + '.jpg'
            csv_writer.writerow([image_path, flagPosition[0], flagPosition[1], flagPosition[2], flagPosition[3], country])


            print('frame count:' + str(frame_cnt))   
            # 等待并判断按键，如果按键为q，退出循环
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    print('执行完毕')

    cap.release()   #关闭视频
    csvfile.close() # 关闭csv文件
    cv2.destroyAllWindows() #关闭窗口





def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def blur_demo(image):  # 均值模糊  去随机噪声有很好的去燥效果
    i = random.randint(0, 1)
    if i == 1:
        dst = cv2.blur(image, (1, 15))  # （1, 15）是垂直方向模糊，（15， 1）还水平方向模糊
    else :
        dst = cv2.blur(image, (15, 1))
    # cv2.namedWindow('blur_demo', cv2.WINDOW_NORMAL)
    # # print(dst.shape)
    # cv2.imshow("blur_demo", dst)
    return dst
#定义添加椒盐噪声的函数
def SaltAndPepper(src,percetage):
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)==0:
            SP_NoiseImg[randX,randY]=0
        else:
            SP_NoiseImg[randX,randY]=255
    return SP_NoiseImg

#定义添加高斯噪声的函数
def addGaussianNoise(image,percetage):
    G_Noiseimg = image
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,image.shape[0]-1)
        temp_y = np.random.randint(0,image.shape[1]-1)
        i = random.randint(0,1)
        if i ==1:
            G_Noiseimg[temp_x][temp_y] = 255
        if i ==0:
            G_Noiseimg[temp_x][temp_y] = 0
    return G_Noiseimg

'''
input：
    src_image:源图片
    threshold:二值化阈值
output：
    binary_image:二值化图片    
'''
def toBinary(src_image, threshold = 15):
    # 显示原始帧
    # cv2.imshow('src_image',src_image)   
    # 灰度
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray_image)
    # 二值化
    ret, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY) # 阈值:15 可调
    # cv2.imshow('binary', binary_image)

    return binary_image


'''
input：
    binary_image:二值化图片  
output：
    maxContour:最大轮廓    
''' 
def findMaxContour(binary_image):
    # 寻找所有轮廓
    image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 轮廓面积
    contour_Area = 0
    # 遍历所有轮廓 寻找面积最大的轮廓
    for i in range(len(contours)):
        contour_Area_temp = cv2.contourArea(contours[i])
        if contour_Area_temp > contour_Area:
            contour_Area = contour_Area_temp
            maxContour = contours[i] # 最大的轮廓

    return maxContour


'''
function:裁剪最小外接矩形图片
input：
    frame:原始帧
    boundRect:最小外接矩形
''' 
def cutOutMinRect(frame, minRect):
    # 确定旋转参数
    if  minRect[1][0] < minRect[1][1]:
        rotationAngle = 90 + minRect[2]
        minRect_height = int(minRect[1][0])
        minRect_width = int(minRect[1][1])
    else:
        rotationAngle = minRect[2]
        minRect_height = int(minRect[1][1])
        minRect_width = int(minRect[1][0])

    minRect_center_x = int(minRect[0][0])
    minRect_center_y = int(minRect[0][1])

    # 最小外接矩形 旋转变换
    rotationMatrix = cv2.getRotationMatrix2D((minRect_center_x, minRect_center_y), rotationAngle, 1)  # 获得旋转矩阵
    rotation_frame = cv2.warpAffine(frame, rotationMatrix, (frame.shape[1], frame.shape[0])) # 按照旋转矩阵进行仿射变换
    # 裁剪最小外接矩形
    minRect_image = rotation_frame[minRect_center_y - minRect_height//2 : minRect_center_y + minRect_height//2, 
                                   minRect_center_x - minRect_width//2 : minRect_center_x + minRect_width//2]

    return minRect_image


'''
function:裁剪外接矩形图片
input：
    frame:原始帧
    boundRect:外接矩形
''' 
def cutOutBoundRect(frame, boundRect):
    boundRect_image = frame[boundRect[1] : boundRect[1] + boundRect[3], 
                            boundRect[0] : boundRect[0] + boundRect[2]]

    return boundRect_image


'''
input：
    src_image:源图片
    boundRect:外接矩形
    minRect:最小外接矩形 
''' 
def drawRect(src_image, boundRect, minRect):
    # 绘制最小外接矩形
    minRect_vertices = cv2.boxPoints(minRect)   # minRect_vertices 最小外接矩形的四个顶点
    minRect_vertices = np.int0(minRect_vertices)
    cv2.drawContours(src_image, [minRect_vertices], 0, (0, 0, 255), 2)
    # 绘制边框矩形
    cv2.rectangle(src_image, (boundRect[0], boundRect[1]), 
                (boundRect[0] + boundRect[2], boundRect[1] + boundRect[3]), (0, 255, 0), 2)
    # 显示标注图像    
    cv2.imshow('mark', src_image)


'''
input：
    background_path:背景图片路径
    background_list:背景路径下的文件列表
    boundRect_image:外接矩形图片
output：
    background_image:背景图片    
''' 
def selectBackground(background_path, background_list, boundRect_image):
    # 每次读取不同的背景
    background_index = random.randint(0, len(background_list) - 1)
    backgroundPic_path = background_path + background_list[background_index]
    background_image = cv2.imread(backgroundPic_path)

    boundRect_rows, boundRect_cols, boundRect_channels = boundRect_image.shape
    background_rows, background_cols, background_channels = background_image.shape
    while boundRect_rows > background_rows or boundRect_cols > background_cols:
        background_index = random.randint(0, len(background_list) - 1)
        backgroundPic_path = background_path + background_list[background_index]
        background_image = cv2.imread(backgroundPic_path)
        boundRect_rows, boundRect_cols, boundRect_channels = boundRect_image.shape
        background_rows, background_cols, background_channels = background_image.shape

    return background_image


'''
function:填充轮廓内部
input：
    frame:原始帧
    maxContour:轮廓
    boundRect:外接矩形边框
output：
    boundRect_image_mask:外接矩形图像掩模    
''' 
def fillContourToMask(frame, maxContour, boundRect):
    contourFilled_image = frame.copy()
    cv2.drawContours(contourFilled_image, [maxContour], 0, (255, 255, 0), -1) # -1为填充, 填充颜色为(255, 255, 0)
    boundRect_image_mask = cutOutBoundRect(contourFilled_image, boundRect)

    return boundRect_image_mask


'''
function:图像缩放和增广
input：
    boundRect_image:外接矩形图像
    boundRect_image_mask:外接矩形图像掩模
output：
    boundRect_image:缩放和增广后的外接矩形图像
    boundRect_image_mask:缩放后的外接矩形图像掩模       
''' 
def imageAugment(boundRect_image, boundRect_image_mask, ratiow, ratioh, number):
    # 随机缩放比例
    if ratiow>=ratioh:
        m = ratioh
    else:
        m = ratiow
    scale = random.uniform(0.5, m)
    boundRect_image = cv2.resize(boundRect_image, (int(boundRect_image.shape[1] * scale), int(boundRect_image.shape[0] * scale)))
    boundRect_image_mask = cv2.resize(boundRect_image_mask, (int(boundRect_image_mask.shape[1] * scale), int(boundRect_image_mask.shape[0] * scale)))
    # 随机镜像翻转
    isFlip = random.randint(0, 1)
    if isFlip == 1:
        boundRect_image = cv2.flip(boundRect_image, 1)
        boundRect_image_mask = cv2.flip(boundRect_image_mask, 1)

    a = random.uniform(0.3, 2)
    b = random.uniform(-3, 3)
    e = random.randint(0, 1)
    if e == 1:
        boundRect_image = Contrast_and_Brightness(a, b, boundRect_image)

    c = random.randint(0, 1)
    d = 0
    if c ==1:
        d = random.randint(0, 1)
    if d ==1:
        boundRect_image = blur_demo(boundRect_image)

    f = random.randint(0, 20)
    if f == 5:
        boundRect_image = addGaussianNoise(boundRect_image, 0.01)
    if f == 10:
        boundRect_image = SaltAndPepper(boundRect_image, 0.1)
    # 图像增广
    g = random.randint(0, 10)
    if g == 1:
        aug_seq = iaa.Sequential([iaa.CoarseDropout(p=0.1, size_percent=0.05, per_channel=0.5)])
        # boundRect_image = boundRect_image.copy()
        boundRect_image = aug_seq.augment_image(boundRect_image)
    # boundRect_image_mask = aug_seq.augment_image(boundRect_image_mask)
    # cv2.imshow('augmentation', boundRect_image_aug)

    return boundRect_image, boundRect_image_mask


'''
input：
    boundRect_image:外接矩形图像
    boundRect_image_mask:外接矩形图像掩模
    background_image:背景图片
output：
    merge_image:合成图片
    flagPosition:合成之后旗帜的位置信息       
'''         
def mergeImage(boundRect_image, boundRect_image_mask, background_image):
    boundRect_rows, boundRect_cols, boundRect_channels = boundRect_image.shape # 矩形框的行列
    background_rows, background_cols, background_channels = background_image.shape # 背景的行列
    # 随机选取贴图的原点
    originPoint = [np.random.randint(0, background_rows - boundRect_rows), np.random.randint(0, background_cols - boundRect_cols)]
    flagPosition = [originPoint[1], originPoint[0], originPoint[1] + boundRect_cols, originPoint[0] + boundRect_rows] # 合成图中旗子的位置信息[x1, y1, x2, y2]
    # 矩阵操作 替换像素 合成图像
    mask = (boundRect_image_mask[:, :, 0] == 255) * (boundRect_image_mask[:, :, 1] == 255) * (boundRect_image_mask[:, :, 2] == 0)
    mask = mask[:, :, None] # 二维变三维
    merge_image = background_image.copy() # 背景的副本
    merge_image[flagPosition[1] : flagPosition[3], flagPosition[0] : flagPosition[2]] = \
        merge_image[flagPosition[1] : flagPosition[3], flagPosition[0] : flagPosition[2]] * ~mask + boundRect_image * mask
    # cv2.rectangle(merge_image, (flagPosition[0], flagPosition[1]), (flagPosition[0] + boundRect_cols, flagPosition[1] + boundRect_rows), (0, 255, 0), 2)
    # cv2.imshow('merge_image', merge_image)

    return merge_image, flagPosition


if __name__ == "__main__":
    main()

