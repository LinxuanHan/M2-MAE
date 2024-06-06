from torch._C import dtype
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk
from .transforms import Center_Crop, Compose,RandomResize,RandomCrop,RandomCrop3D
from torch.utils.data import Dataset as dataset
import random

class Test_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = os.listdir(self.args.test_data_path)

        self.transforms = Compose([
            RandomCrop(self.args.crop_size),
            RandomResize(128, 240, 240),
        ])


    def __getitem__(self, index):
        ct = sitk.ReadImage(os.path.join(self.args.test_data_path,self.filename_list[index]))
        # seg = sitk.ReadImage(self.filename_list[index][1][:-7]+"_seg"+self.filename_list[index][1][-7:], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        # ct_array = np.transpose(ct_array,[3,0,1,2])

        max_v = np.max(ct_array[0])
        ct_array[0] = (ct_array[0]) / max_v
        max_v = np.max(ct_array[1])
        ct_array[1] = (ct_array[1]) / max_v
        max_v = np.max(ct_array[2])
        ct_array[2] = (ct_array[2]) / max_v
        max_v = np.max(ct_array[3])
        ct_array[3] = (ct_array[3]) / max_v
        ct_array = (ct_array - 0.5) * 2


        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array)
        random_num = random.randint(0,3)
        seg_array = torch.zeros([155, 240, 240])
        # ct_array[random_num] = torch.zeros([155, 240, 240])

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array.unsqueeze(0))
        seg_array = torch.zeros([128, 240, 240])
        seg_array[:] = ct_array[random_num]
        ct_array[random_num] = torch.zeros([128, 240, 240])

        return ct_array, seg_array, random_num

    def __len__(self):
        return len(self.filename_list)


class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取一个data文件并归一化 、resize
        self.ct = sitk.ReadImage(data_path)
        self.data_np = sitk.GetArrayFromImage(self.ct)
        print(self.data_np.shape)
        # self.data_np = np.transpose(self.data_np, [3, 0, 1, 2])
        # self.data_np = ndimage.zoom(self.data_np, (1, 128 / 155, 128 / 240, 128 / 240), order=3)
        self.ori_shape = self.data_np.shape
        print(self.ori_shape)

        # max_v = np.max(self.data_np)
        # min_v = np.min(self.data_np)
        # self.data_np = (self.data_np - min_v) / (max_v - min_v)
        # self.data_np = (self.data_np - 0.5) * 2
        max_v = np.max(self.data_np[0])
        self.data_np[0] = (self.data_np[0]) / max_v
        max_v = np.max(self.data_np[1])
        self.data_np[1] = (self.data_np[1]) / max_v
        max_v = np.max(self.data_np[2])
        self.data_np[2] = (self.data_np[2]) / max_v
        max_v = np.max(self.data_np[3])
        self.data_np[3] = (self.data_np[3]) / max_v
        self.data_np = (self.data_np - 0.5) * 2
        print(np.max(self.data_np))

        self.resized_shape = self.data_np.shape
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        self.data_np = self.padding_img(self.data_np, self.cut_size,self.cut_stride)
        self.padding_shape = self.data_np.shape
        # 对数据按步长进行分patch操作，以防止显存溢出
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)
        # print(self.data_np.shape)
        # 读取一个label文件 shape:[s,h,w]
        self.seg = sitk.ReadImage(label_path,sitk.sitkInt8)
        self.label_np = sitk.GetArrayFromImage(self.seg)
        self.label_np = ndimage.zoom(self.label_np, (128/155, 128 / 240, 128 / 240), order=0)
        if self.n_labels==2:
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()

        # 预测结果保存
        self.result = None

    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])

        data = torch.FloatTensor(data)#.unsqueeze(0)
        # print("data:",data.shape)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_result(self):

        patch_s = self.result.shape[2]

        N_patches_img = (self.padding_shape[1] - patch_s) // self.cut_stride + 1
        assert (self.result.shape[0] == N_patches_img)
        full_prob = torch.zeros((self.n_labels, self.padding_shape[1], self.ori_shape[2],self.ori_shape[3]))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros((self.n_labels, self.padding_shape[1], self.ori_shape[2], self.ori_shape[3]))

        for s in range(N_patches_img):
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.ori_shape[1], :self.ori_shape[2], :self.ori_shape[3]]
        return img.unsqueeze(0)

    def padding_img(self, img, size, stride):
        # print(img.shape)
        assert (len(img.shape) == 4)  # 3D array
        img_c, img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((img_c, s, img_h, img_w),dtype=np.float32)
        tmp_full_imgs[:,:img_s] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_c, img_s, img_h, img_w = img.shape
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1

        print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, img_c, size, img_h, img_w), dtype=np.float32)

        for s in range(N_patches_img):  # loop over the full images
            patch = img[:,s * stride : s * stride + size]
            patches[s] = patch
        print(patches.shape)
        return patches  # array with all the full_imgs divided in patches

def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'data/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath,args=args), datapath.split('\\')[-1]
