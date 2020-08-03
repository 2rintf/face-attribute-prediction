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


BATCH_SIZE = 64
EPOCH = 50
ROOT_PATH = "/home/czd-2019/Projects/celebA_dataset"


def get_error_num(model_pred,labels,threshold=0.6):
    pred_result = model_pred > threshold
    pred_result = pred_result.float()
    r,l = pred_result.size()
    # print(pred_result)
    print(pred_result.shape)
    print(labels.shape)
    # 得到预测值与标签值不一致的类的个数
    temp = pred_result-labels
    error = temp[temp!=0]
    error_num = len(error)
    # error_rate = error_num*1.0/(r*l*1.0)
    # # print(error_rate)
    return  error_num




def calculate_acuracy_mode_one(model_pred, labels):
    """
    计算准确率——方式1

    设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签.

    注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为是这一类的概率.

    预测结果大于这个阈值则视为预测正确.
    :param model_pred: 预测值
    :param labels: 标签
    :return: precision.item(), recall.item()
    """
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    print(pred_result)
    pred_one_num = torch.sum(pred_result)
    print(pred_one_num)
    # 这个if的前提是，每一个数据必定有一个标签分类。但是我的情况是，有可能出现样本
    # 不属于任何一个标签。
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    print(target_one_num)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item(), recall.item()

def calculate_acuracy_mode_two(model_pred, labels):
    """
    计算准确率——方式2

    取预测概率最大的前top个标签，作为模型的预测结果。

    取前top个预测结果作为模型的预测结果
    :param model_pred:
    :param labels:
    :return:
    """
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0, pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall



def adjust_lr(optimizer, epoch, maxepoch, init_lr, power = 0.9):
    lr = init_lr * (1-epoch/maxepoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))

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

def main():
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

    val_loader = torch.utils.data.DataLoader(
        CelebA(ROOT_PATH, '20_val_data.txt', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        CelebA(ROOT_PATH, '20_test_data.txt', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True)

    model = AttrPre()
    model.to(device)
    init_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # Multi-label 选择使用BCE损失函数。注意pytorch的BCELoss要求:
    # ① 必须经过sigmoid()
    # ② label是FloatTensor
    loss_func = nn.BCEWithLogitsLoss().to(device)
    print(model)

    model.train()
    train_loss = 0.0
    for epoch in range(EPOCH):
        for i, (input, target) in enumerate(train_loader):

            optimizer.zero_grad()

            print(input.shape)
            labels = get_each_attr_label(target)

            hair_color,hair_cut,sex,beard,skin,eyes = model(input.to(device))
            # print(hair_color.cpu().detach().numpy().shape)
            # print(labels[0].numpy().shape)
            print(sex.shape)
            print(labels[2].shape)
            print(sex)
            print(labels[2].unsqueeze(1).type(torch.FloatTensor))# 从[64]拉成[64, 1]，以匹配输入的维数

            # 提取各分类器各自的label, 并且转为FloatTensor.
            label1_f = labels[0].type(torch.FloatTensor).to(device)
            label2_f = labels[1].type(torch.FloatTensor).to(device)
            label3_f = labels[2].unsqueeze(1).type(torch.FloatTensor).to(device)
            label4_f = labels[3].type(torch.FloatTensor).to(device)
            label5_f = labels[4].unsqueeze(1).type(torch.FloatTensor).to(device)
            label6_f =labels[5].type(torch.FloatTensor).to(device)

            # 计算个分类器Loss
            loss_haircolor = loss_func(hair_color,label1_f)
            loss_haircut = loss_func(hair_cut,label2_f)
            loss_sex = loss_func(sex,label3_f)
            loss_beard = loss_func(beard,label4_f)
            loss_skin = loss_func(skin,label5_f)
            loss_eyes = loss_func(eyes,label6_f)

            # todo print loss

            # todo 考虑6个loss间的权重？
            total_loss = loss_haircolor+loss_haircut+loss_sex+loss_beard+loss_skin+loss_eyes
            total_loss.backward()

            # 累计错误个数
            error1 = get_accuracy(torch.sigmoid(hair_color), label1_f)
            error2 = get_accuracy(torch.sigmoid(hair_cut), label1_f)
            error3 = get_accuracy(torch.sigmoid(sex), label1_f)
            error4 = get_accuracy(torch.sigmoid(beard), label1_f)
            error5 = get_accuracy(torch.sigmoid(skin), label1_f)
            error6 = get_accuracy(torch.sigmoid(hair), label1_f)


            exit(0)

            optimizer.step()

        # todo every epoch print accuracy

        # todo save checkpoint





if __name__ == '__main__':
    main()


