from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger,common
from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
from models import ResUNet,SeNet,ECAResNet,UNETR,TransBTS,UNETR_Multi,UNETR_MultiCat,UNETR_3DMAE,Cross_UNETR
from utils.metrics import DiceAverage
from collections import OrderedDict


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            print("output",output.shape)
            # output = output[:,:,:155,]
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())
    print(img_dataset.result.shape)
    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred,dim=1)

    pred_img = common.to_one_hot_3d(pred,args.n_labels)
    print("pred_img",pred_img.shape)
    test_dice.update(pred_img, target)
    
    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    # if args.n_labels==3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    
    pred = np.asarray(pred.numpy(),dtype='uint8')
    if args.postprocess:
        pass # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred,axis=0))

    return test_dice, pred

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # device = torch.device('cpu')
    # model info
    model = UNETR.UNETR(
        in_channels=4,
        out_channels=4,
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
    # model = Cross_UNETR.UNETR(
    #     in_channels=1,
    #     out_channels=4,
    #     img_size=(64, 240, 240),
    #     feature_size=16,
    #     hidden_size=384,
    #     mlp_dim=1024,
    #     num_heads=12,
    #     pos_embed='conv',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=True,
    #     dropout_rate=0.2).to(device)
    # model = TransBTS.TransBTS(in_channel=4, out_channel=4,training=True).to(device)
    # model = ResUNet(in_channel=4, out_channel=4, training=True).to(device)
    # model = ECAResNet.ResUNet(in_channel=2, out_channel=4, training=True).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    print(save_path)
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    datasets = Test_Datasets(args.test_data_path,args=args)
    for img_dataset,file_idx in datasets:

        test_dice,pred_img = predict_one_img(model, img_dataset, args)

        test_log.update(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx))
