#!coding=utf-8  
#####################################
# 对训练集合进行可视化
#####################################
import cv2
import pandas as pd

#在图上画框 
def showPicResult(image, coords, name_set):  
    img = cv2.imread(image)  
    for i in range(len(coords)):  
        x1=coords[i][0] - coords[i][2]/2
        y1=coords[i][1] - coords[i][3]/2
        x2=coords[i][0] + coords[i][2]/2
        y2=coords[i][1] + coords[i][3]/2
        name = name_set[i]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3) 
        cv2.putText(img,name,(int(x1),int(y1+20)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.namedWindow("image_detector", 1)    #1表示原图
    cv2.moveWindow("image_detector",0,0)
    cv2.resizeWindow("image_detector", 256,192) #可视化的图片大小
    cv2.imshow('image_detector', img)  
    cv2.waitKey(0)    #表示等待500ms，0表示一直等待直到按键
    cv2.destroyAllWindows()  

#转化为预测的框格式
def transfer(coords):
    '''
    输入：  
        coords：坐标，形式为"x_min_y_min_x_max_y_max"
    输出：
        转换后的坐标：(x,y,w,h) x和y分别为中心横纵坐标，w为框的宽度，h为高度
    '''
    coords = coords.split('_')
    x_min = float(coords[0])
    y_min = float(coords[1])
    x_max = float(coords[2])
    y_max = float(coords[3])
    
    transfered_x = (x_min + x_max)/2
    transfered_y = (y_min + y_max)/2
    transfered_w = x_max - x_min
    transfered_h = y_max - y_min
    
    return (transfered_x,transfered_y,transfered_w,transfered_h)
      
if __name__ == "__main__":  
    import os
    txt_path = './txt'
    pic_root_path = './train'
    for parent, _, files in os.walk(txt_path):
        for file in files:
            pic_file = open(os.path.join(parent, file), 'r')
            ori_img = pic_root_path+'/'+file.split('.')[0]+'.jpg'
            coords_set = []
            name_set = []
            for line in pic_file.readlines():
                name = line.split(' ')[0]
                coord = line.split(' ')[1]+'_'+line.split(' ')[2]+'_'+line.split(' ')[3]+'_'+line.split(' ')[4]
                coords_set.append(transfer(coord))
                name_set.append(name)
            print(file)
            showPicResult(ori_img, coords_set, name_set)


