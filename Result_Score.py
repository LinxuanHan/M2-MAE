# -*- coding: utf-8 -*-

import numpy as np
import os
import SimpleITK as sitk


def DiceScore(predict, label):
    predict_num = sitk.GetArrayFromImage(predict)
    label_num = sitk.GetArrayFromImage(label)
    predict_score = np.sum(predict_num)
    label_score = np.sum (label_num)
    and_score = np.sum(np.bitwise_and(predict_num,label_num))
    dice_score = 2*and_score/(predict_score+label_score)
    # print(dice_score)
    return dice_score


if __name__ == "__main__":
    label_path = "F:\\3DUNet-Pytorch-master_0314\\raw_dataset\\test\\label"
    predict_path = "F:\\3DUNet-Pytorch-master_0314\\raw_dataset\\test\\final"
    label_path_list = os.listdir(label_path)
    predict_path_list = os.listdir(predict_path)
    dice_list = []
    for i in range(len(label_path_list)):
        label_path_i = os.path.join(label_path,label_path_list[i])
        predict_path_i = os.path.join(predict_path,predict_path_list[i])
        label_i = sitk.ReadImage(label_path_i)
        predict_i = sitk.ReadImage(predict_path_i)
        dice = DiceScore(predict_i, label_i)
        dice_list.append(dice)

    print(dice_list)
    print(np.mean(dice_list))
    print(np.var(dice_list))

