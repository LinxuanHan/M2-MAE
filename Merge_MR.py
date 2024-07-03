import SimpleITK as sitk
import os
import numpy as np
import shutil
def Merge_Imge(item):
    file = r'C:\Users\HLX\Desktop\BraTS2023_show_train'
    # file_list = os.listdir(file)
    print(file_list)
    # seg_file = os.path.join(file, "data_t1c")
    t1c_file = os.path.join(file, "data_t1c",item)
    t1n_file = os.path.join(file, "data_t1n",item)
    t2f_file = os.path.join(file, "data_t2f",item)
    t2w_file = os.path.join(file, "data_t2w",item)

    # seg_nii = sitk.ReadImage(seg_file)
    t1c_nii = sitk.ReadImage(t1c_file)
    t1n_nii = sitk.ReadImage(t1n_file)
    t2f_nii = sitk.ReadImage(t2f_file)
    t2w_nii = sitk.ReadImage(t2w_file)

    # seg_num = sitk.GetArrayFromImage(seg_nii)
    t1c_num = sitk.GetArrayFromImage(t1c_nii)
    t1n_num = sitk.GetArrayFromImage(t1n_nii)
    t2f_num = sitk.GetArrayFromImage(t2f_nii)
    t2w_num = sitk.GetArrayFromImage(t2w_nii)

    tmp = np.zeros((64,240,240,4))
    # tmp = t2w_num
    # tmp[tmp>max] = max
    tmp[:, :, :, 0] = t1c_num[:64,:]
    tmp[:, :, :, 1] = t1n_num[:64,:]
    tmp[:, :, :, 2] = t2f_num[:64,:]
    tmp[:, :, :, 3] = t2w_num[:64,:]
    tmp = tmp.astype(int)
    tmp_nii = sitk.GetImageFromArray(tmp)
    sitk.WriteImage(tmp_nii, os.path.join(r"C:\Users\HLX\Desktop\BraTS2023_show_train\data",item))
    # shutil.copy(seg_file,os.path.join(r"C:\Users\HLX\Desktop\BraTS2023_show_train\label",item+".nii.gz"))
    # print(seg_num.shape)


if __name__ == "__main__":
    file = r"C:\Users\HLX\Desktop\BraTS2023_show_train\label"
    file_list = os.listdir(file)
    for i in range(len(file_list)):
        Merge_Imge(file_list[i])


