import pandas as pd
import os
import shutil

target_root_path = './txt_for_merge'
if os.path.exists(target_root_path):
    shutil.rmtree(target_root_path)
os.mkdir(target_root_path)

len_list = list()
for i in range(1,5):
    csv_path = './res'+str(i)+'.csv'
    if i==1:
        data = pd.read_csv(csv_path)
    else:
        data_temp = pd.read_csv(csv_path)
        data = pd.concat([data, data_temp])
    len_list.append(len(data['filename']))

# print(len_list)
# print(len(data['filename']))
data.to_csv('./noIndex.csv', index=False)
data = pd.read_csv('./noIndex.csv')

index = -1
indexx = 0
xishu = [1, 1, 1, 1]    #加权系数
for i in range(len(data['filename'])):
    pic_name = data['filename'][i]
    label = data['label'][i]
    if i%len_list[indexx]==0:
        index += 1
        indexx += 1
    score = float(data['score'][i]) * xishu[index]
    x_min = data['x_min'][i]
    y_min = data['y_min'][i]
    x_max = data['x_max'][i]
    y_max = data['y_max'][i]
    txt_path = os.path.join(target_root_path, pic_name.split('.')[0]+'.txt')
    with open(txt_path, 'a+') as f:
        content = str(label)+' '+str(score)+ ' '+str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)+'\n'
        f.write(content)
    print('write to txt '+str(i+1)+'/'+str(len(data['filename'])))

print('done')
