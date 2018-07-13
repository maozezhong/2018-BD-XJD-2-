import os
import pandas as pd
from nms import nms
import numpy as np
import cv2

csv_path = '/home/maozezhong/Desktop/baidu_fusai/data/class_name.csv'
data = pd.read_csv(csv_path)
label2name = dict()
for i in range(len(data['label'])):
    label2name[int(data['label'][i])] = data['prefix'][i]

def showPicResult(image, coords):  
    img = cv2.imread(image)  
    for i in range(len(coords)):  
        x1=coords[i][0]
        y1=coords[i][1]
        x2=coords[i][2]
        y2=coords[i][3]
        score = coords[i][4]
        score = round(score,2)
        label = coords[i][5]
        name = label2name[label] + ' ' + str(score)
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3) 
        cv2.putText(img,name,(int(x1),int(y1+20)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.namedWindow("retinanet_image_detector", 0)  #1的时候是原图
    cv2.moveWindow("retinanet_image_detector",0,0)
    cv2.resizeWindow("retinanet_image_detector", 640, 960);
    cv2.imshow('retinanet_image_detector', img)  
    cv2.waitKey(0)    #表示等待500ms，0表示一直等待直到按键
    cv2.destroyAllWindows()

file_names = list()
labels_ = list()
scores_ = list()
x_mins = list()
y_mins = list()
x_maxs = list()
y_maxs = list()

txt_path = './txt_for_merge'
test_txt_path = '/home/maozezhong/Desktop/baidu_fusai/data/datasets/test.txt'
with open(test_txt_path, 'r') as f:
    for line in f.readlines():
        pic_name = line.strip()
        img_path = os.path.join('/home/maozezhong/Desktop/baidu_fusai/data/datasets/test',pic_name)
        txt_pic_path = os.path.join(txt_path, pic_name.split('.')[0]+'.txt')
        dets = list()
        with open(txt_pic_path, 'r') as ff:
            for ll in ff.readlines():
                ll = ll.strip()
                label = int(ll.split(' ')[0])
                score = float(ll.split(' ')[1])
                x_min = int(ll.split(' ')[2])
                y_min = int(ll.split(' ')[3])
                x_max = int(ll.split(' ')[4])
                y_max = int(ll.split(' ')[5])
                dets.append([x_min,y_min,x_max,y_max,score,label])
        dets = np.array(dets)
        dets = nms(dets)
        for det in dets:
            file_names.append(pic_name)
            labels_.append(int(det[5]))
            scores_.append(det[4])
            x_mins.append(int(det[0]))
            y_mins.append(int(det[1]))
            x_maxs.append(int(det[2]))
            y_maxs.append(int(det[3]))

        # 可视化
        # showPicResult(img_path, dets)

column = ['filename', 'label', 'score', 'x_min', 'y_min', 'x_max', 'y_max']
dataframe = pd.DataFrame({'filename': file_names, 'label': labels_, 'score' : scores_, 'x_min' : x_mins, 'y_min' : y_mins, 'x_max' : x_maxs, 'y_max' : y_maxs})
dataframe.to_csv('./res_merge.csv', index=False, header=False, columns=column, sep=' ') 