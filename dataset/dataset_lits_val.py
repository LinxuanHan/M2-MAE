from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Center_Crop, Compose,RandomResize,RandomCrop,RandomCrop3D
import random


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))

        self.transforms = Compose([
            RandomCrop(self.args.crop_size),
            # RandomResize(128, 128, 128),
            RandomResize(128, 240, 240),
            # RandomResize(128, 240, 240),
        ])


    def __getitem__(self, index):
        ct = sitk.ReadImage(self.filename_list[index][0])
        seg = sitk.ReadImage(self.filename_list[index][1][:-7]+self.filename_list[index][1][-7:], sitk.sitkUInt8)
        # seg = sitk.ReadImage(self.filename_list[index][1][:-7]+"_seg"+self.filename_list[index][1][-7:], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        # ct_array = np.transpose(ct_array,[3,0,1,2])
        seg_array = sitk.GetArrayFromImage(seg)

        max_v = np.max(ct_array[0])
        ct_array[0] = (ct_array[0]) / max_v
        max_v = np.max(ct_array[1])
        ct_array[1] = (ct_array[1]) / max_v
        max_v = np.max(ct_array[2])
        ct_array[2] = (ct_array[2]) / max_v
        max_v = np.max(ct_array[3])
        ct_array[3] = (ct_array[3]) / max_v
        ct_array = (ct_array - 0.5) * 2

        # max_v = np.max(ct_array)
        # ct_array = (max_v - ct_array)/ max_v
        # ct_array = (ct_array - 0.5) * 2


        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        random_num = random.randint(0,3)
        seg_array[:] = ct_array[random_num]
        ct_array[random_num] = torch.zeros([self.args.crop_size, 240, 240])

        return ct_array, seg_array#, random_num

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Dataset(args, mode='train')

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())