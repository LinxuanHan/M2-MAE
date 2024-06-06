import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
if __name__ == "__main__":
    path = r"F:\HLX\2DUnet\experiments\2DResUNet_ALL_MASK_3\result"
    path_list = os.listdir(path)
    for i in path_list:
        array = Image.open(os.path.join(path,i)).convert("L")
        # A=np.array([[4,3,2,4],[5,4,7,8],[9,16,11,5],[13,3,4,16],[6,18,1,20]])
        plt.matshow(array, cmap=plt.cm.Reds)
        plt.title("matrix A")
        plt.show()
