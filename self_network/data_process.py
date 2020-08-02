import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path



attr_path="test_40_att_list.txt"
root = "/home/czd-2019/Projects/celebA_dataset"
haircolor_attr =[9,10,12,18]
hair_attr = [5,6,29,33,34]
sex_attr = [21]
beard_attr=[17,23,25]
skin_attr = [27]
eye_attr = [2,4,13,16,23,24]

# [9, 10, 12, 18, 5, 6, 29, 33, 34, 21, 17, 23, 25, 27, 2, 4, 13, 16, 23, 24]
total_label=haircolor_attr+hair_attr+sex_attr+beard_attr+skin_attr+eye_attr


# file = open("./filter_celeba_data.txt",'w')

filter_img_list = []
filter_label_list = []

count = 0
for line in open(os.path.join(root,attr_path),'r'):
    sample = line.split()
    if len(sample)!=41:
        print("File maybe errors. Not 40 attribute.")
        continue
    img_t = sample[0]
    label_t = [int(i) for i in sample[1:]]
    for h in total_label:
        # if label_t[h] == 1:
        filter_img_list.append(img_t)
        # 把label拿出来
        label_t = [label_t[i] for i in total_label]
        label_t = [1 if j == 1 else 0 for j in label_t]
        filter_label_list.append(label_t)
        break


print(len(filter_img_list))
print(len(filter_label_list))

file = open("./data_list/20_test_data.txt",'w')
for img,label in zip(filter_img_list,filter_label_list):
    label = [str(i) for i in label]
    file.write(img+" ")
    for l in label:
        file.write(l+" ")
    file.write("\n")


file.close()
