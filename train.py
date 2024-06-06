from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from functools import partial

import torch
import torch.nn as nn
from models import models_mae,UNETR_Multi,UNETR_MultiCat,UNETR, MAE3D,MAE3D_Gen

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
from copy import copy,deepcopy

def val(model, val_loader, loss_func):
    model.eval()
    with torch.no_grad():
        for idx,(data,target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data ,target = data.float(),target.float()
            data ,target = data.to(device),target.to(device)
            output = model(data,target)
            loss = loss_func(output)

    val_log = OrderedDict({'Val_Loss': loss})
    return val_log

def train(model, train_loader, optimizer, loss_func):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data,target)
        loss = loss_func(output)
        print(loss)
        loss.backward()
        optimizer.step()

    val_log = OrderedDict({'Train_Loss': loss})
    return val_log

def para_state_dict(model, model_save_path):
    state_dict = deepcopy(model.state_dict())
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        for key in state_dict:  # 在新的网络模型中遍历对应参数
            if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
                print("成功初始化参数:", key)
                state_dict[key] = loaded_paras[key]
    return state_dict

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args),batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=8,num_workers=args.n_threads, shuffle=False)

    # model info
    # model = UNETR.UNETR(
    #     in_channels=4,
    #     out_channels=1,
    #     img_size=(128, 240, 240),
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=True,
    #     dropout_rate=0.2).to(device)
    model = MAE3D_Gen.MaskedAutoencoderViT(
        img_size=(128, 240, 240),
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    # model = models_mae.mae_vit_base_patch16_dec512d8b().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    # para_state_dict(model,r'F:\HLX\3DSelf_Training\experiments\Multi_self_train_random\best_model.pth')

    if args.continue_train == True:
        ckpt = torch.load('{}\\best_model.pth'.format(save_path))
        model.load_state_dict(ckpt['net'])
    loss = loss.MAELoss()
    #
    log = logger.Train_Logger(save_path,"train_log")

    best = [0,10000] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4 # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss)
        val_log = val(model, val_loader, loss)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_Loss'] < best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_Loss']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
    #     torch.cuda.empty_cache()