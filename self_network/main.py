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


BATCH_SIZE = 32
EPOCH = 50
ROOT_PATH = "/home/czd-2019/Projects/celebA_dataset"




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

def get_each_attr_label(labels):
    # [9, 10, 12, 18, 5, 6, 29, 33, 34, 21, 17, 23, 25, 27, 2, 4, 13, 16, 23, 24]
    target = np.array(labels)
    # print(target)
    haircolor_label = target[:,0:4]
    haircut_label = target[:,4:9]
    sex_label = [target[:,9]]
    beard_label = [target[:,i] for i in range(10,13)]
    skin_label = [target[:,13]]
    eyes_label = [target[:,i] for i in range(14,20)]
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
    loss_func = nn.CrossEntropyLoss().to(device)
    print(model)

    for epoch in range(EPOCH):
        for i, (input, target) in enumerate(train_loader):
            print(input.shape)
            labels = get_each_attr_label(target)

            hair_color,hair_cut,sex,beard,skin,eyes = model(input.to(device))

            loss_haircolor = loss_func(hair_color,labels[0].to(device))
            loss_haircut = loss_func(hair_cut,labels[1].to(device))
            loss_sex = loss_func(sex,labels[2].to(device))
            loss_beard = loss_func(beard,labels[3].to(device))



            # loss1 = loss_func()






if __name__ == '__main__':
    main()


