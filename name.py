import os

names = os.listdir('/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/gapsab/train/0.normal/')  # 路径
# names1 = os.listdir('/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/gapsab/test/1.abnormal/')  # 路径
i = 0  # 用于统计文件数量是否正确，不会写到文件里
j=0
train_val = open('/home/wyy/PycharmProjects/ABD1/Outlier_Dataset/gapsab/train.txt', 'w')
# train_val1 = open('/home/yy/semi/CCT-master/dataloaders/roadgf/train/1500_train_unsupervised.txt', 'w')
for name in names:
    index = name.rfind('.')
    # print(index)
    name = name[:index]
    # print(name[-1])
    # g1='e'
    # g2='b'
    # if g1 in name[-1] or g2 in name[-1]:#
    train_val.write('/train/0.normal/' + name + '.jpg' + ' ' +'\n')

    # train_val.write('/train/' + name +'.jpg'  + '\n')
    i = i + 1
    # else:
    #    train_val.write('/train/' + name+'.jpg'+ ' '+'0'+ '\n')
    #    j = j + 1

print(i)
# for name1 in names1:
#     index1 = name1.rfind('.')
#     # print(index)
#     name1 = name1[:index1]
#     # print(name[-1])
#     # g1='e'
#     # g2='b'
#     # if g1 in name[-1] or g2 in name[-1]:#
#     train_val.write('/test/1.abnormal/' + name1 + '.jpg' + ' ' + '1' + '\n')
#
#     # train_val.write('/train/' + name +'.jpg'  + '\n')
#     j = j + 1
#     # else:
#     #    train_val.write('/train/' + name+'.jpg'+ ' '+'0'+ '\n')
#     #    j = j + 1
#
# print(j)
# print(j)