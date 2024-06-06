from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models import models_mae,UNETR_ViT_Backbone,UNETR

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk
from dataset.dataset_lits_test import Test_Dataset


def val(model, val_loader, loss_func):
    model.eval()
    val_loss = metrics.LossAverage()
    with torch.no_grad():
        for idx,(data) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data  = data.float()
            data  = data.to(device)
            output = model(data)
            loss = loss_func(output)

            val_loss.update(loss.item(), data.size(0))
    # print(val_loss)
    val_log = OrderedDict({'Val_Loss': val_loss.avg})
    return val_log

def train(model, train_loader, optimizer, loss_func):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    for idx, (data) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.float()
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_func(output)
        print(loss)
        loss.backward()
        optimizer.step()

    val_log = OrderedDict({'Train_Loss': loss})
    return val_log


def predict(model, test_loader, result_save_file):
    model.eval()

    with torch.no_grad():
        for idx, (data, target,random_num) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # print(idx,file_name[0])
            print(data.shape,target.shape,random_num)

            result = model(data)
            print("result",result.shape)
            result = result.cpu().detach().numpy()[0][0]
            print("result",result.shape)
            target = target.cpu().detach().numpy()[0]
            # print(target.astype())
            result_image = sitk.GetImageFromArray(result)
            target_image = sitk.GetImageFromArray(target)
            sitk.WriteImage(result_image,os.path.join(result_save_file,str(idx)+"_"+str(random_num.cpu().detach().numpy()[0])+".nii.gz"))
            sitk.WriteImage(target_image,os.path.join(result_save_file,"target_"+str(idx)+"_"+str(random_num.cpu().detach().numpy()[0])+".nii.gz"))

if __name__ == '__main__':
    torch.manual_seed(3407)
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    # model = UNETR_ViT_Backbone.UNETR(
    #     in_channels=1,
    #     out_channels=1,
    #     img_size=(64, 240, 240),
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=True,
    #     dropout_rate=0.2).to(device)
    model = UNETR.UNETR(
        in_channels=4,
        out_channels=1,
        img_size=(128, 240, 240),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.2).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load('{}\\best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    test_loader = DataLoader(dataset=Test_Dataset(args), batch_size=1, num_workers=1, shuffle=False)
    predict(model, test_loader, result_save_path)