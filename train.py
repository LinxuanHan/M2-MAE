from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models import UNet, ResUNet , KiUNet_min, SegNet, SeNet, DResUNet, UNETR, TransBTS,\
    ECAResNet,TransBTS_ECA,UNETR_Multi,UNETR_MultiCat,UNETR_3DMAE,Cross_UNETR

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_1': val_dice.avg[1]})
    if n_labels==4:
        val_log.update({'Val_dice_2': val_dice.avg[2]})
        val_log.update({'Val_dice_3': val_dice.avg[3]})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        # print(target.shape)
        target = common.to_one_hot_3d(target,n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("OK")
        output = model(data)
        # print(output.shape)
        loss = loss_func(output,target)
        # loss0 = loss_func(output[0], target)
        # loss1 = loss_func(output[1], target)
        # loss2 = loss_func(output[2], target)
        # loss3 = loss_func(output[3], target)
        # loss4 = loss_func(output[4], target)

        # loss = loss1 + alpha * loss0
        # loss = loss4 +  alpha * (loss0 + loss1 + loss2 +loss3)
        # loss = loss3 + alpha * (loss0 + loss1 + loss2)
        loss.backward()

        optimizer.step()
        # print(loss)
        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output, target)
        # train_dice.update(output[3], target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_1': train_dice.avg[1]})
    if n_labels==4:
        val_log.update({'Train_dice_2': train_dice.avg[2]})
        val_log.update({'Train_dice_3': train_dice.avg[3]})
    return val_log

if __name__ == '__main__':
    torch.manual_seed(3407)
    args = config.args
    save_path = os.path.join('.\experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args),batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=1,num_workers=args.n_threads, shuffle=False)
    # model = ECAResNet.ResUNet(in_channel=4, out_channel=4, training=True).to(device)
    model = UNETR.UNETR(
        in_channels=4,
        out_channels=4,
        img_size=(64,240,240),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.2).to(device)

    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    if args.continue_train == True:
        ckpt = torch.load('{}\\best_model.pth'.format(save_path))
        model.load_state_dict(ckpt['net'])
    print("OK")
    # loss = loss.Mixup()
    loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4 # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_1'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_1']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰减
        # if epoch % 100 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()    