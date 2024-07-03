import SimpleITK as sitk
from PIL import Image
import numpy as np
import os


def Get_Multi_ROI(file_path, label_path):
    Image = sitk.ReadImage(file_path)
    Label = sitk.ReadImage(label_path)
    img_num = sitk.GetArrayFromImage(Image)
    label_num = sitk.GetArrayFromImage(Label)
    print(img_num.shape)
    print(label_num.shape)
    # for i in range(len(img_num)):
    #     img_num[i][label_num == 0] = 0
    # print(img_num.shape)
    # img_num = img_num.transpose([1, 2, 3, 0])
    # Image = sitk.GetImageFromArray(img_num)
    # sitk.WriteImage(Image, r"ROI.nii")
    # print(img_num.shape)


def Write_Train_PNG(path,file_out_path,file_path):
    path = os.path.join(path,file_path)
    Img = sitk.ReadImage(path)
    img_num = sitk.GetArrayFromImage(Img)
    # max_num = np.max(img_num)
    # min_num = np.min(img_num)
    # print(np.max(img_num))
    # print(np.min(img_num))
    # img_num = (img_num - min_num) / (max_num-min_num)
    # img_num = img_num*255
    # print(np.max(img_num))
    # print(np.min(img_num))
    for i in range(len(img_num)):
        image_array = img_num[i]
        max_num = np.max(image_array)
        min_num = np.min(image_array)
        print(np.max(image_array))
        print(np.min(image_array))
        image_array = (image_array - min_num) / (max_num-min_num)
        image_array = image_array*255
        print(np.max(image_array))
        print(np.min(image_array))
        for j in range(len(img_num[i])):
            im = Image.fromarray(np.uint8(image_array[j]))
            im = im.convert('L')
            out_path = os.path.join(file_out_path,file_path.replace(".nii.gz", "_" + str(i) + "_" + str(j) + ".png"))
            im.save(out_path)

def Write_Test_RNG(path,file_out_path,label_path):
    path = os.path.join(path,label_path)
    Img = sitk.ReadImage(path)
    img_num = sitk.GetArrayFromImage(Img)
    # img_num = (img_num/3)*255
    for j in range(len(img_num)):
        im = Image.fromarray(np.uint8(img_num[j]))
        im = im.convert('L')
        out_path = os.path.join(file_out_path,label_path.replace(".nii.gz", "_" + str(j) + ".png"))
        im.save(out_path)


file_path = r"F:\Task01_BrainTumour\preprocess\data"
label_path = r"F:\Task01_BrainTumour\preprocess\label"
file_out_path = r"F:\Task01_BrainTumour\preprocess\data_png"
label_out_path = r"F:\Task01_BrainTumour\preprocess\label_png"

if __name__ == "__main__":
    file = label_path
    file_path_list = os.listdir(file)
    count = 0
    for i in file_path_list:
        # count += 1
        # if count<346:
        #     continue
        file_name = os.path.join(file, i)
        print(file_name)
        # Get_Multi_ROI(file_name,label_name)
        # Write_Train_PNG(file,label_out_path,i)
        Write_Test_RNG(file,label_out_path,i)
