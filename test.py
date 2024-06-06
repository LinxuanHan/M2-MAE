from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger,common
from dataset.dataset_lits_test import Test_Dataset
from utils.common import  to_one_hot_3d
import os
import numpy as np
from models import models_mae_test
from utils.metrics import DiceAverage
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
def plot_img(pre,img,id):
    print(id[0])
    for i in range(len(id[0])):
        if id[0][i] == 0:
            continue
        else:
            raw = i//15
            line = i%15
            print(raw,line)
            img[0][0][raw*16:raw*16+16,line*16:line*16+16] = pre[0][0][raw*16:raw*16+16,line*16:line*16+16]
            img[0][1][raw * 16:raw * 16 + 16, line * 16:line * 16 + 16] = pre[0][1][raw * 16:raw * 16 + 16,line * 16:line * 16 + 16]
            img[0][2][raw * 16:raw * 16 + 16, line * 16:line * 16 + 16] = pre[0][2][raw * 16:raw * 16 + 16,line * 16:line * 16 + 16]
            img[0][3][raw * 16:raw * 16 + 16, line * 16:line * 16 + 16] = pre[0][3][raw * 16:raw * 16 + 16,line * 16:line * 16 + 16]
    plt.matshow(img[0][0], cmap=plt.cm.gray)
    plt.title("matrix A")
    plt.show()
    plt.matshow(img[0][1], cmap=plt.cm.gray)
    plt.title("matrix A")
    plt.show()
    plt.matshow(img[0][2], cmap=plt.cm.gray)
    plt.title("matrix A")
    plt.show()
    plt.matshow(img[0][3], cmap=plt.cm.gray)
    plt.title("matrix A")
    plt.show()


def predict(model, test_loader,  result_save_file):
    model.eval()
    test_log = logger.Test_Logger(save_path, "test_log")
    
    with torch.no_grad():
        for idx,(data,file_name) in tqdm(enumerate(test_loader),total=len(test_loader)):
            # print(idx,file_name[0])
            data = data.float()
            data = data.to(device)
            loss,pred_img,ids_restore,mask,imgs = model(data)
            print(pred_img.shape)
            print(pred_img.cpu().detach().numpy()[0])
            print(ids_restore)
            print(mask)
            plt.matshow(pred_img.cpu().detach().numpy()[0][0], cmap=plt.cm.gray)
            plt.title("matrix A")
            plt.show()
            plt.matshow(pred_img.cpu().detach().numpy()[0][1], cmap=plt.cm.gray)
            plt.title("matrix A")
            plt.show()
            plt.matshow(pred_img.cpu().detach().numpy()[0][2], cmap=plt.cm.gray)
            plt.title("matrix A")
            plt.show()
            plt.matshow(pred_img.cpu().detach().numpy()[0][3], cmap=plt.cm.gray)
            plt.title("matrix A")
            plt.show()
            plot_img(pred_img.cpu().detach().numpy(),imgs.cpu().detach().numpy(),mask.cpu().detach().numpy())

            # plt.matshow(imgs.cpu().detach().numpy()[0][0], cmap=plt.cm.gray)
            # plt.title("matrix A")
            # plt.show()
            # plt.matshow(imgs.cpu().detach().numpy()[0][1], cmap=plt.cm.gray)
            # plt.title("matrix A")
            # plt.show()
            # plt.matshow(imgs.cpu().detach().numpy()[0][2], cmap=plt.cm.gray)
            # plt.title("matrix A")
            # plt.show()
            # plt.matshow(imgs.cpu().detach().numpy()[0][3], cmap=plt.cm.gray)
            # plt.title("matrix A")
            # plt.show()

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join(r'F:\HLX\2DMAE\experiments', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = models_mae_test.mae_vit_base_patch16_dec512d8b().to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    # data info
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    test_loader = DataLoader(dataset=Test_Dataset(args),batch_size=1,num_workers=1,shuffle=False)
    predict(model,test_loader,result_save_path)

