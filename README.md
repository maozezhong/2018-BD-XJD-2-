## 比赛介绍
- 针对检测+分类任务，我们提供9000张带有位置信息和类别信息的图像数据用于训练，4351张图像用于评估测试。该数据集全部来源于百度地图淘金，选取了60类常见品牌类别。比如，肯德基，星巴克，耐克等。

## 比赛历程
- 7月13号结束,7月初开始做.中间尝试了
    - [yolo](https://github.com/pjreddie/darknet),
    - [faster-rcnn pytorch版本](https://github.com/jwyang/faster-rcnn.pytorch)
    - [dcn](https://github.com/msracver/Deformable-ConvNets)
    - [retinanet keras版本](https://github.com/fizyr/keras-retinanet)
    - fpn等
- 最后还是tow-stage的faster-rcnn正确度高(至少在我的实验中是这样,当然时间有限,设备有限单卡1080,实验结论不完备)
- 数据处理方面 : 使用了针对检测的数据增强,包括旋转,平移,加噪,改亮度,具体实现见DataAugmentForObejctDetection.py这个脚本
- trick方面 : 1)softnms, 2)模型融合(具体见merge_box中的脚本)
- batchsize基本上是1,设备受限上不去了; lr初始一般设的0.001, 每5轮降为原来的十分之一; 输入尺度试过600和800
- 最后线上为0.8576,排名23,没苟进决赛,哎...

## 脚本说明:
- merge_box:
    - csv_2_txt_for_merge.py : 根据结果csv产生中间txt文件
    - merge_res.py : 融合并产生最终csv结果文件
- show_boundingbox_on_pic.py : 可视化脚本
- DataAugmentForObejctDetection.py : 数据增强脚本
- densenet.py : pytorch, 基于densenet backbone的faster rcnn模型结构(未实验), 参考[vision/torchvision/models/densenet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
- resnext.py : pytorch, 基于resnext backbone的faster rcnn模型结构, 参考[ResNeXt-PyTorch/resnext.py](https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py), 其实就是在resnet的基础上加了多通道并行.
- nms.py : 常规nms,以及softnms

## 学习姿势
- faster**多尺度训练,多尺度预测**,**tesnsorlayer数据增强**,结果ensemble可以到89
- 基于fpn的faster-rcnn
- detectron单模型可以到89
- predict的时候augment, detectron中的config文件下有例子
- 调参方法,何凯明论文,detctron有论文链接
- **增强后的数据作为验证集**,把60类验证集AP保存,取每一类最好的ap的模型进行集成
- 数据增强库 : [imgaug](https://blog.csdn.net/u012897374/article/details/80142744), emmm...应该比我自己整的靠谱点
- 调整分类和bbox的loss权重
- 使用sniper模型
- ssd上89!!!但是没说用了啥技巧....

## 所有代码链接
- [baiduyun(part1:models部分)](https://pan.baidu.com/s/1BaXyPzJkpRCMlsC2saDnnA)
- [baiduyun(part2:data)](https://pan.baidu.com/s/1k9E_KsEtz5f0lbzX_2OjBg)

## to do list
- [x] 使用detectron
- [X] [transfer to coco data](https://blog.csdn.net/qq_15969343/article/details/80848175)
- [ ] test aug
- [ ] 数据增强,[sampleParing and mixup](https://kexue.fm/)

