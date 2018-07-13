# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     maozezhong 2018-6-27
##############################################################

# 包括:
#     1. 裁剪(需改变bbox)
#     2. 平移(需改变bbox)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
# 注意:   
#     random.seed(),相同的seed,产生的随机数是一样的!!

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def show_pic(img, bboxes, names):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        name = names[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
        cv2.putText(img,name,(int(x_min),int(y_min+20)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5, 
                crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                add_noise_rate=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
    
    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time())) 
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True)*255

    
    # 调整亮度
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5) #flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)
    
    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        #---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        #---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])
        
        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        #---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
        d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离

        #随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        #确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        #---------------------- 裁剪boundingbox ----------------------
        #裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min])
        
        return crop_img, crop_bboxes
  
    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        #---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   #裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #包含所有目标框的最大左移动距离
        d_to_right = w - x_max      #包含所有目标框的最大右移动距离
        d_to_top = y_min            #包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max     #包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])  #x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        #---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y])

        return shift_img, shift_bboxes

    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  #改变的次数
        # print('------')
        # random.seed(int(time.time()))
        if random.random() < self.crop_rate:        #裁剪
            # print('裁剪')
            change_num += 1
            img, bboxes = self._crop_img_bboxes(img, bboxes)

        if random.random() > self.rotation_rate:    #旋转
            # print('旋转')
            change_num += 1
            angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            scale = random.uniform(0.7, 0.8)
            img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
        
        if random.random() < self.shift_rate:        #平移
            # print('平移')
            change_num += 1
            img, bboxes = self._shift_pic_bboxes(img, bboxes)
        
        if random.random() > self.change_light_rate: #改变亮度
            # print('亮度')
            change_num += 1
            img = self._changeLight(img)

        if random.random() < self.add_noise_rate:    #加噪声
            # print('加噪声')
            change_num += 1
            img = self._addNoise(img)
        # print('------')
        return img, bboxes, change_num
            

if __name__ == '__main__':
    import shutil
    # test
    dataAug = DataAugmentForObjectDetection()
    agument_num = 1
    source_txt_root_path = '/home/maozezhong/Desktop/baidu_fusai/data/datasets/txt'
    source_pic_root_path = '/home/maozezhong/Desktop/baidu_fusai/data/datasets/train'
    target_pic_root_path = './data_augment/JPEGImages'
    target_txt_root_path = './data_augment/txt'
    if os.path.exists(target_pic_root_path):
        shutil.rmtree(target_pic_root_path)
    if os.path.exists(target_txt_root_path):
        shutil.rmtree(target_txt_root_path)
    os.mkdir(target_pic_root_path)
    os.mkdir(target_txt_root_path)
    cnt = 0
    for parent, _, files in os.walk(source_txt_root_path):
        for file in files:
            cnt += 1
            pic_path = os.path.join(source_pic_root_path, file.split('.')[0]+'.jpg')
            txt_path = os.path.join(parent, file)
            txt_file = open(txt_path, 'r')
            contents = txt_file.readlines()
            bboxes = list()
            names = list()
            for content in contents:
                content = content.strip()
                x_min = int(float(content.split(' ')[1]))
                y_min = int(float(content.split(' ')[2]))
                x_max = int(float(content.split(' ')[3]))
                y_max = int(float(content.split(' ')[4]))
                bboxes.append([x_min, y_min, x_max, y_max])
                names.append(content.split(' ')[0])
            img = cv2.imread(pic_path)
            # 原图可视化
            # show_pic(img, bboxes, names)

            i = 0
            while i < agument_num:

                # 数据增强后的图
                changed_img, changed_bboxes, change_num = dataAug.dataAugment(img, bboxes)
                # show_pic(changed_img, changed_bboxes, names)

                #必须得有一个改变
                if change_num == 0:
                    continue
                i += 1
                
                # 写入txt
                target_txt_path = os.path.join(target_txt_root_path, file.split('.')[0]+'_'+str(i)+'.txt')
                target_txt_file = open(target_txt_path, 'w')
                for ii in range(len(changed_bboxes)):
                    bbox = changed_bboxes[ii]
                    content = names[ii] + ' ' + str(int(bbox[0]))+' ' + str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n'
                    target_txt_file.write(content)
                # 写入pic
                target_pic_path = os.path.join(target_pic_root_path, file.split('.')[0]+'_'+str(i)+'.jpg')
                cv2.imwrite(target_pic_path, changed_img)

            print(str(cnt)+'/'+str(len(files)))
        print('done!')



