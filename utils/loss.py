"""
基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""

import torch
import torch.nn as nn

from utils import Hausdorff

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)

class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard  += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)

class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:,i] - target[:,i]).pow(2) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target[:,i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:,i] - target[:,i]).pow(2) * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)

class TverskyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)+
                        0.3 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)
class HausdorffLoss(nn.Module):
    def __init__(self, p=2):
        super(HausdorffLoss, self).__init__()
        self.p = p

    def torch2D_Hausdorff_distance(self, x, y, p=2):  # Input be like (Batch,1, width,height) or (Batch, width,height)
        x = x.float()
        y = y.float()
        distance_matrix = torch.cdist(x, y, p=p)  # p=2 means Euclidean Distance

        value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
        value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

        value = torch.cat((value1, value2), dim=1)

        return value.max(1)[0].mean()

    def forward(self, x, y):  # Input be like (Batch,height,width)
        loss = self.torch2D_Hausdorff_distance(x, y, self.p)
        return loss
class Mixup(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        hdLoss1 = 0.
        smooth = 1
        dice = 0.

        for i in range(pred.size(1)):
            HD1 = Hausdorff.HausdorffLoss()
            hdLoss1 += HD1(pred[:, i].unsqueeze(1),
                          target[:,i].unsqueeze(1))
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        hdLoss1 = hdLoss1 / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2) + torch.clamp(hdLoss1.mean(), 0, 2)