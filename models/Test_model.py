"""
This code is referenced from https://github.com/assassint2017/MICCAI-LITS2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2 ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(4,16,3,1,1),
        )

    def forward(self, inputs):
        output = self.encoder_stage1(inputs)
        return output

if __name__ == "__main__":
    import config
    args = config.args
    device = torch.device('cpu')
    model = ResUNet(in_channel=4, out_channel=4,training=True).to(device)
    input = torch.ones([1,4,224,224]).to(device)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)

    output = model(input)
    print(output.shape)