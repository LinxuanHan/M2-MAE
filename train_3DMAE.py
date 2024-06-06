from dataset.dataset_3DMAE_val import Val_Dataset
from dataset.dataset_3DMAE_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from functools import partial

import torch
import torch.nn as nn
from models import models_mae,UNETR_ViT_Backbone,MAE3D

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict


def val(model, val_loader, loss_func):
    model.eval()
    val_loss = metrics.LossAverage()
    with torch.no_grad():
        for idx,(data) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data = data.float()
            data = data.to(device)
            output = model(data)
            loss = loss_func(output)

            val_loss.update(loss.item(), data.size(0))
    # print(val_loss)
    val_log = OrderedDict({'Val_Loss': val_loss.avg})
    return val_log

def train(model,D,train_loader, optimizer_G,optimizer_D, loss_G,loss_D):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer_G.state_dict()['param_groups'][0]['lr']))
    model.train()
    D.train()
    for idx, (data) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.float()
        data = data.to(device)

        batch_size = data.size()[0]
        y_real = torch.ones(batch_size).to(device)
        y_fake = torch.zeros(batch_size).to(device)

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        D_result = D(data).squeeze()
        D_real_loss = loss_D(D_result,y_real)
        print(D_real_loss)
        loss_mae, output = model(data)
        loss_mae = loss_mae.detach()
        output = output.detach()
        D_result = D(output).squeeze()
        D_fake_loss = loss_D(D_result,y_fake)
        print(D_fake_loss)
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimizer_D.step()

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_mae, output = model(data)
        print(loss_mae)
        D_result = D(output).squeeze()
        G_train_loss = loss_D(D_result.detach(), y_real) + loss_G(loss_mae)
        print(G_train_loss)
        G_train_loss.backward()
        optimizer_G.step()


    val_log = OrderedDict({'Train_Loss': G_train_loss})
    return val_log

if __name__ == '__main__':
    torch.manual_seed(3407)
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=16, num_workers=args.n_threads, shuffle=False)

    # model info
    model = UNETR_ViT_Backbone.UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(64, 240, 240),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.2).to(device)
    D = UNETR_ViT_Backbone.discriminiator().to(device)

    # model = MAE3D.MaskedAutoencoderViT(
    #     img_size=(128, 240, 240),
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)

    optimizer_G = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer_D = optim.AdamW(model.parameters(), lr=args.lr)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    if args.continue_train == True:
        ckpt = torch.load('{}\\best_model.pth'.format(save_path))
        model.load_state_dict(ckpt['net'])

    loss_G = loss.MAELoss()
    loss_D = nn.BCELoss()
    #
    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 10000]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer_G, epoch, args)
        common.adjust_learning_rate(optimizer_D, epoch, args)
        train_log = train(model,D,train_loader, optimizer_G,optimizer_D, loss_G,loss_D)
        # val_log = val(model,D val_loader, loss_G,loss_D)
        # log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        # trigger += 1
        # if val_log['Val_Loss'] < best[1]:
        #     print('Saving best model')
        #     torch.save(state, os.path.join(save_path, 'best_model.pth'))
        #     best[0] = epoch
        #     best[1] = val_log['Val_Loss']
        #     trigger = 0
        # print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
    #     if epoch % 30 == 0: alpha *= 0.8
    #
    #     # early stopping
    #     if args.early_stop is not None:
    #         if trigger >= args.early_stop:
    #             print("=> early stopping")
    #             break
    # #     torch.cuda.empty_cache()