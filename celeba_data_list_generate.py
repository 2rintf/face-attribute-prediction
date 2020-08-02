import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path



attr_path="list_attr_celeba.txt"
generate_root = "./"

file = open("./train_40_att_list.txt",'w')

count = 0
change= False
for line in open(attr_path,'r'):

    if count==162770 :
        file.close()
        file=open("./val_40_att_list.txt",'w')
    elif count==182637:
        file.close()
        file =open("test_40_att_list.txt",'w')

    sample = line.split()
    if len(sample)!=41:
        print("File maybe errors. Not 40 attribute.")
        continue

    count+=1
    file.writelines(line)

file.close()

images = []
targets = []
file_check=open("./train_40_att_list.txt",'r')
for line in file_check:
    sample = line.split()
    if len(sample) != 41:
        raise (RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
    images.append(sample[0])
    targets.append([int(i) for i in sample[1:]])
print(len(images))
print(len(targets))
file_check.close()

images = []
targets = []
file_check=open("./val_40_att_list.txt",'r')
for line in file_check:
    sample = line.split()
    if len(sample) != 41:
        raise (RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
    images.append(sample[0])
    targets.append([int(i) for i in sample[1:]])
print(len(images))
print(len(targets))
file_check.close()

images = []
targets = []
file_check=open("./test_40_att_list.txt",'r')
for line in file_check:
    sample = line.split()
    if len(sample) != 41:
        raise (RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
    images.append(sample[0])
    targets.append([int(i) for i in sample[1:]])
print(len(images))
print(len(targets))
file_check.close()