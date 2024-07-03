import numpy
import SimpleITK
import os
import SimpleITK as sitk
import numpy as np
import cv2
from skimage.measure import label


def limit(data, max, min):
    data[data > max] = max
    data[data < min] = min
    return data


def preprocess_item(raw_path, result_path):
    img_dt = sitk.ReadImage(raw_path)
    origin = img_dt.GetOrigin()
    direction = img_dt.GetDirection()
    space = img_dt.GetSpacing()
    img_data = sitk.GetArrayFromImage(img_dt)
    if len(img_data) < 48:
        return

    # if len(img_data[0])>512:
    #     return
    # if len(img_data[0][0])>512:
    #     return

    img_data = np.clip(img_data, 0, 8000)

    savedImg = sitk.GetImageFromArray(img_data)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, result_path)


def preprocess_Center(raw_path, result_path):
    for patient in os.listdir(raw_path):
        patient = os.path.join(raw_path, patient)
        for ct in os.listdir(patient):
            # print(ct)
            if "axial" not in ct.split("_"):
                print(ct)
                continue
            if "diff" in ct.split("_"):
                print(ct)
                continue
            if "t1" not in ct.split("_"):
                print(ct)
                continue

            if ct.split(".")[-1] == "nii":
                preprocess_item(os.path.join(patient, ct), os.path.join(result_path, ct))


def resize_item(raw_path, ct_path):
    img_dt = sitk.ReadImage(raw_path)
    origin = img_dt.GetOrigin()
    direction = img_dt.GetDirection()
    space = img_dt.GetSpacing()
    img_data = sitk.GetArrayFromImage(img_dt)

    l = len(img_data[0])
    w = len(img_data[0][0])
    n = l if l > w else w
    print(n, l, w)
    expand_data = np.zeros((len(img_data), n, n))
    expand_data[:, (n - len(img_data[0])) // 2:(n - len(img_data[0])) // 2 + len(img_data[0]),
    (n - len(img_data[0][0])) // 2:(n - len(img_data[0][0])) // 2 + len(img_data[0][0])] = img_data

    savedImg = sitk.GetImageFromArray(expand_data)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, ct_path)


def resize_Center(raw_path, ct_path):
    for ct in os.listdir(raw_path):
        resize_item(os.path.join(raw_path, ct),
                    os.path.join(ct_path, ct))


def expand_item(raw_path, ct_path, label_path):
    img_dt = sitk.ReadImage(raw_path)
    origin = img_dt.GetOrigin()
    direction = img_dt.GetDirection()
    space = img_dt.GetSpacing()
    img_data = sitk.GetArrayFromImage(img_dt)

    expand_data = np.zeros((len(img_data), 512, 512))
    for i in range(len(img_data)):
        expand_data[i] = cv2.resize(img_data[i], (512, 512), interpolation=cv2.INTER_NEAREST)
    # expand_data[expand_data<0] = 0
    # expand_data[expand_data>200] = 200
    # expand_data = expand_data - 100
    # expand_data = expand_data * 2

    savedImg = sitk.GetImageFromArray(expand_data)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, ct_path)

    label = np.zeros((len(img_data), 512, 512))

    savedImg = sitk.GetImageFromArray(label)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, label_path)


def expand_Center(raw_path, ct_path, label_path):
    for ct in os.listdir(raw_path):
        expand_item(os.path.join(raw_path, ct),
                    os.path.join(ct_path, ct),
                    os.path.join(label_path, ct))
        print(ct + " is finished!")


def standard_item(raw_path, result_path):
    img_dt = sitk.ReadImage(raw_path)
    origin = img_dt.GetOrigin()
    direction = img_dt.GetDirection()
    space = img_dt.GetSpacing()
    img_data = sitk.GetArrayFromImage(img_dt)

    standard_dt = sitk.ReadImage(result_path)
    standard_data = sitk.GetArrayFromImage(standard_dt)

    savedImg = sitk.GetImageFromArray(standard_data)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, result_path)


def standard_result(raw_path, result_path):
    for ct in os.listdir(raw_path):
        print(ct)
        result = os.path.join(result_path, "result-" + ct.split(".")[0] + ".nii.gz")
        if os.path.exists(result):
            standard_item(os.path.join(raw_path, ct), result)
            print(result + " standard is finished ！")
        else:
            print(ct)


def max_connectzoom_item(raw_path, final_result):
    raw_data = sitk.ReadImage(raw_path)
    connectzoom = sitk.ConnectedComponent(raw_data)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(connectzoom, raw_data)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    result = sitk.GetArrayFromImage(connectzoom)
    outmask = result.copy()
    outmask[result == maxlabel] = 1
    outmask[result != maxlabel] = 0
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(raw_data.GetSpacing())
    outmasksitk.SetOrigin(raw_data.GetOrigin())
    outmasksitk.SetDirection(raw_data.GetDirection())
    sitk.WriteImage(outmasksitk, final_result)


def max_connectzoom(raw_path, final_result):
    for ct in os.listdir(raw_path):
        print(ct)
        max_connectzoom_item(os.path.join(raw_path, ct), os.path.join(final_result, ct))
        print(ct + " max_zoom is finished!")


def padding_item(bw, hole_min, hole_max):
    fill_out = np.zeros_like(bw)
    for zz in range(bw.shape[0]):
        background_lab = label(~bw[zz, :, :], connectivity=1)  # 1表示4连通， ~bw[zz, :, :]1变为0， 0变为1
        # 标记背景和孔洞， target区域标记为0
        out = np.copy(background_lab)
        component_sizes = np.bincount(background_lab.ravel())
        # 求各个类别的个数
        too_big = component_sizes > hole_max
        too_big_mask = too_big[background_lab]
        out[too_big_mask] = 0

        too_small = component_sizes < hole_min
        too_small_mask = too_small[background_lab]
        out[too_small_mask] = 0
        # 大于最大孔洞和小于最小孔洞的都标记为0， 所以背景部分被标记为0了。只剩下符合规则的孔洞
        fill_out[zz, :, :] = out
        # 只有符合规则的孔洞区域是1， 背景及target都是0
    return np.logical_or(bw, fill_out)  # 或运算，孔洞的地方是1，原来target的地方也是1


def padding(raw_path, final_result):
    for ct in os.listdir(raw_path):
        print(ct)
        img_dt = sitk.ReadImage(os.path.join(raw_path, ct))
        origin = img_dt.GetOrigin()
        direction = img_dt.GetDirection()
        space = img_dt.GetSpacing()
        img_data = sitk.GetArrayFromImage(img_dt)

        standard_data = padding_item(img_data, 0, 10000)
        result = standard_data.astype(np.int)
        savedImg = sitk.GetImageFromArray(result)

        savedImg.SetOrigin(origin)
        savedImg.SetDirection(direction)
        savedImg.SetSpacing(space)
        sitk.WriteImage(savedImg, os.path.join(final_result, ct))
        print(ct + " padding is finished!")


if __name__ == "__main__":
    path = "F:\\UNet-3D-master(last)\\center\\nii\\metastasis\\MR"
    path_preprocess = "F:\\UNet-3D-master(last)\\center_preprocess"
    center_standard = "F:\\UNet-3D-master(last)\\center_standard"
    center_ct = "F:\\UNet-3D-master(last)\\UNet-3D-master\\raw_dataset\\center\\ct"
    center_label = "F:\\UNet-3D-master(last)\\UNet-3D-master\\raw_dataset\\center\\label"
    result_path = "F:\\3DUNet-Pytorch-master_0314\\experiments\\220402\\result"
    final_result = "F:\\UNet-3D-master(last)\\UNet-3D-master\\raw_dataset\\center\\final"

    # preprocess_Center(path,path_preprocess)
    #
    # resize_Center(path_preprocess,center_standard)
    #
    # expand_Center(center_standard,center_ct,center_label)

    # center_ct = "F:\\3DUNet-Pytorch-master_0314\\raw_dataset\\test\\ct"
    # result_path = "F:\\3DUNet-Pytorch-master_0314\\experiments\\ResUNet\\result"
    # standard_result(center_ct, result_path)

    final_result = result_path
    max_connectzoom(result_path, final_result)
    padding(final_result, final_result)

    # for path_1 in os.listdir(path):
    #     print(path_1)liangwei
    #     path_1 = os.path.join(path,path_1)
    #     for path_2 in os.listdir(path_1):
    #         if path_2.split(".")[-1] == "nii":
    #             outpath_1 = os.path.join(outpath,path_2.split(".")[0]+"_expand.nii")
    #             expand_item(os.path.join(path_1,path_2),outpath_1)