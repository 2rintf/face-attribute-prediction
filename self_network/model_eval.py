import argparse
import torch
# import torch.utils.data
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from self_network.celeba import CelebA
from self_network.model_res import *

import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join, basename, dirname
from os import makedirs, remove
import shutil
from torch.optim import lr_scheduler
import math
import cv2 as cv
import matplotlib.pyplot as plt



BATCH_SIZE = 256
EPOCH = 80
ROOT_PATH = "/home/czd-2019/Projects/celebA_dataset"
pic_path="./pic/"

def get_error_num(model_pred,labels,threshold=0.6):
    pred_result = model_pred > threshold
    pred_result = pred_result.float()
    r,l = pred_result.size()
    # print(pred_result.shape)
    # print(labels.shape)
    # 得到预测值与标签值不一致的类的个数
    temp = pred_result-labels
    error = temp[temp!=0]
    error_num = len(error)
    # error_rate = error_num*1.0/(r*l*1.0)
    # # print(error_rate)
    return  error_num



def get_each_attr_label(target):
    # [9, 10, 12, 18, 5, 6, 29, 33, 34, 21, 17, 23, 25, 27, 2, 4, 13, 16, 24]
    # target = np.array(labels)
    # print(target)
    haircolor_label = target[:,0:4]
    haircut_label = target[:,4:9]
    sex_label = target[:,9]
    beard_label = target[:,10:13]
    skin_label = target[:,13]
    eyes_label = target[:,14:20]
    return haircolor_label,haircut_label,sex_label,beard_label,skin_label,eyes_label


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])


train_dataset = CelebA(
    ROOT_PATH,
    '20_train_data.txt',
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True)

val_dataset = CelebA(
    ROOT_PATH,
    '20_val_data.txt',
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True)

test_dataset = CelebA(
    ROOT_PATH,
    '20_test_data.txt',
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True)

model = AttrPre()
model.to(device)
checkpoint = torch.load("/home/czd-2019/Projects/face-attribute-prediction/self_network/checkpoint/checkpoint_epoch40.pth")
model.load_state_dict(checkpoint['model_state_dict'])

init_lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=init_lr)
loss_func = nn.BCEWithLogitsLoss().to(device)
print(model)

model.eval()
total_error = 0
running_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        labels = get_each_attr_label(target)

        hair_color, hair_cut, sex, beard, skin, eyes = model(input.to(device))

        # 提取各分类器各自的label, 并且转为FloatTensor.
        label1_f = labels[0].type(torch.FloatTensor).to(device)
        label2_f = labels[1].type(torch.FloatTensor).to(device)
        label3_f = labels[2].unsqueeze(1).type(torch.FloatTensor).to(device)
        label4_f = labels[3].type(torch.FloatTensor).to(device)
        label5_f = labels[4].unsqueeze(1).type(torch.FloatTensor).to(device)
        label6_f = labels[5].type(torch.FloatTensor).to(device)

        # 计算个分类器Loss
        loss_haircolor = loss_func(hair_color, label1_f)
        loss_haircut = loss_func(hair_cut, label2_f)
        loss_sex = loss_func(sex, label3_f)
        loss_beard = loss_func(beard, label4_f)
        loss_skin = loss_func(skin, label5_f)
        loss_eyes = loss_func(eyes, label6_f)

        running_loss[1] += loss_haircolor
        running_loss[2] += loss_haircut
        running_loss[3] += loss_sex
        running_loss[4] += loss_beard
        running_loss[5] += loss_skin
        running_loss[6] += loss_eyes

        # todo 由于数据正负样本的不平衡，是否考虑6个loss间的权重？
        total_loss = loss_haircolor + loss_haircut + loss_sex + loss_beard + loss_skin + loss_eyes
        running_loss[0] += total_loss

        # print loss
        print_loss_step = 10
        if i % print_loss_step == 0 and i != 0:
            print(
                'EVAL: [%5d] total_loss: %.3f \n     | loss1:%.3f  loss2:%.3f  loss3:%.3f loss4:%.3f  loss5:%.3f  loss6:%.3f'
                % ( i + 1, running_loss[0] / print_loss_step,
                   running_loss[1] / print_loss_step,
                   running_loss[2] / print_loss_step,
                   running_loss[3] / print_loss_step,
                   running_loss[4] / print_loss_step,
                   running_loss[5] / print_loss_step,
                   running_loss[6] / print_loss_step))
            running_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 累计错误个数
        error1 = get_error_num(torch.sigmoid(hair_color), label1_f)
        error2 = get_error_num(torch.sigmoid(hair_cut), label2_f)
        error3 = get_error_num(torch.sigmoid(sex), label3_f)
        error4 = get_error_num(torch.sigmoid(beard), label4_f)
        error5 = get_error_num(torch.sigmoid(skin), label5_f)
        error6 = get_error_num(torch.sigmoid(eyes), label6_f)
        total_error += error1 + error2 + error3 + error4 + error5 + error6
    # every epoch print accuracy
    print(len(test_dataset))
    all_num = len(test_dataset) * 20
    epoch_acc = 1 - total_error * 1.0 / all_num
    print('Eval accuracy: %.3f %%' % (epoch_acc))
