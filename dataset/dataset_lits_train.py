from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, RandomResize,RandomCrop3D

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                # RandomResize(128,240,240),
                # RandomResize(64,240,240),#(64,256,256)UNETR
                # RandomResize(128, 240, 240),#raw ECA
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0])
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        # seg = sitk.ReadImage(self.filename_list[index][1][:-7]+"_seg"+self.filename_list[index][1][-7:], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        # ct_array = ct_array[:,:,:,1]
        # print(ct_array.shape)
        # tmp = np.zeros([1,ct_array.shape[0],ct_array.shape[1],ct_array.shape[2]])
        # tmp[0] = ct_array
        # ct_array = tmp
        ct_array = np.transpose(ct_array,[3,0,1,2])
        seg_array = sitk.GetArrayFromImage(seg)
        # print(ct_array.shape)

        ct_array = ct_array.astype(np.float32)
        max_v = np.max(ct_array[0])
        ct_array[0] = (ct_array[0]) / max_v
        max_v = np.max(ct_array[1])
        ct_array[1] = (ct_array[1]) / max_v
        max_v = np.max(ct_array[2])
        ct_array[2] = (ct_array[2]) / max_v
        max_v = np.max(ct_array[3])
        ct_array[3] = (ct_array[3]) / max_v
        # ct_array = (ct_array -0.5)*2


        ct_array = torch.FloatTensor(ct_array)
        # print(ct_array.shape)
        # seg_array[seg_array==2] = 0
        # seg_array[seg_array==3] = 2
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        # seg_array = torch.FloatTensor(seg_array[:64,:]).unsqueeze(0)

        if self.transforms:
            ct_array,seg_array = self.transforms(ct_array, seg_array)
        # print(ct_array.shape,seg_array.shape)
        # print(ct_array.shape,seg_array.squeeze(0).shape)
        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split(' '))
        # print(file_name_list)
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())