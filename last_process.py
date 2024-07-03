import SimpleITK as sitk
import os
import numpy as np

if __name__ == "__main__":
    file_path = r"F:\HLX\3DUNet-BraTS2023\experiments\BraTS2022_2"
    normal_path = os.path.join(file_path, "result_normal")
    ud_path = os.path.join(file_path, "result_UD")
    ud_ur_path = os.path.join(file_path, "result_UD_UR")
    ur_path = os.path.join(file_path, "result_UR")
    ul_path = os.path.join(file_path, "result_UL")
    result_path = os.path.join(file_path, "result_merge")
    list_dir = os.listdir(normal_path)
    for i in range(len(list_dir)):
        normal_file = os.path.join(normal_path,list_dir[i])
        normal_image = sitk.ReadImage(normal_file)
        normal_num = sitk.GetArrayFromImage(normal_image)
        # normal_image = sitk.GetImageFromArray(normal_num)
        # sitk.WriteImage(normal_image,"normal.nii")

        ud_file = os.path.join(ud_path, list_dir[i])
        ud_image = sitk.ReadImage(ud_file)
        ud_num = sitk.GetArrayFromImage(ud_image)
        ud_num = np.flip(ud_num,0)
        # ud_image = sitk.GetImageFromArray(ud_num)
        # sitk.WriteImage(ud_image, "ud.nii")

        ud_ur_file = os.path.join(ud_ur_path, list_dir[i])
        ud_ur_image = sitk.ReadImage(ud_ur_file)
        ud_ur_num = sitk.GetArrayFromImage(ud_ur_image)
        ud_ur_num = np.flip(ud_ur_num, 1)
        ud_ur_num = np.flip(ud_ur_num, 0)
        # ud_ur_image = sitk.GetImageFromArray(ud_ur_num)
        # sitk.WriteImage(ud_ur_image, "ud_ur.nii")

        ur_file = os.path.join(ur_path, list_dir[i])
        ur_image = sitk.ReadImage(ur_file)
        ur_num = sitk.GetArrayFromImage(ur_image)
        ur_num = np.flip(ur_num, 1)
        # ur_image = sitk.GetImageFromArray(ur_num)
        # sitk.WriteImage(ur_image, "ur.nii")

        ul_file = os.path.join(ul_path, list_dir[i])
        ul_image = sitk.ReadImage(ul_file)
        ul_num = sitk.GetArrayFromImage(ul_image)
        ul_num = np.flip(ul_num, 2)
        # ul_image = sitk.GetImageFromArray(ul_num)
        # sitk.WriteImage(ul_image, "ul.nii")

        result_num = np.zeros([2,155,240,240])
        one_hot = np.eye(2)[normal_num]
        one_hot = np.transpose(one_hot, [3, 0, 1, 2])
        result_num += one_hot
        one_hot = np.eye(2)[ud_num]
        one_hot = np.transpose(one_hot, [3, 0, 1, 2])
        result_num += one_hot
        one_hot = np.eye(2)[ur_num]
        one_hot = np.transpose(one_hot, [3, 0, 1, 2])
        result_num += one_hot
        one_hot = np.eye(2)[ud_ur_num]
        one_hot = np.transpose(one_hot, [3, 0, 1, 2])
        result_num += one_hot
        one_hot = np.eye(2)[ul_num]
        one_hot = np.transpose(one_hot, [3, 0, 1, 2])
        result_num += one_hot

        label = np.argmax(result_num, axis=0)
        label = label.astype(int)
        # one_hot = np.transpose(label, [1, 2, 3, 0])
        one_hot = sitk.GetImageFromArray(label)
        sitk.WriteImage(one_hot,os.path.join(result_path,list_dir[i]))
        # print(one_hot.shape)
        print(normal_file)
        # break
