import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import math

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size//2

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,padding=padding,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,l,h,w = x.size()
        avg = self.avg_pool(x).view([b,1,c])

        out = self.conv(avg)

        out = self.sigmoid(out).view([b,c,1,1,1])

        return out*x

class ViT_module(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            img_size,
            feature_size,
            hidden_size,
            mlp_dim,
            num_heads,
            pos_embed="conv",
            norm_name="instance",
            conv_block=False,
            res_block=True,
            dropout_rate=0
    ) -> None:

        super(ViT_module,self).__init__()
        self.patch_size = (feature_size,feature_size,feature_size)
        self.num_layers = 12
        self.classification = False
        self.hidden_size = hidden_size
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=128,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2],hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        _, hidden_states_out = self.vit(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states_out[-1], self.hidden_size, self.feat_size))
        return enc2


class TransBTS(nn.Module):
    def __init__(self, in_channel=1, out_channel=2 ,training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.Attention1 = ECA(16)

        self.Attention2 = ECA(32)

        self.Attention3 = ECA(64)

        self.Attention4 = ECA(128)

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            # nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            # nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            # nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            # nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.vit = ViT_module(
            in_channels=128,
            out_channels=4,
            img_size=(8,32,32),
            feature_size=4,
            hidden_size=128,
            mlp_dim= 512,
            num_heads= 8,
            pos_embed= "perceptron",
            conv_block= False,
            res_block= True,
            dropout_rate= 0.2,
        )

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs)
        attention1 = self.Attention1(long_range1)
        short_range1 = self.down_conv1(attention1)

        long_range2 = self.encoder_stage2(short_range1)+ short_range1
        attention2 = self.Attention2(long_range2)
        long_range2 = attention2 + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)
        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        attention3 = self.Attention3(long_range3)
        long_range3 = attention3 + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)
        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3)
        attention4 = self.Attention4(long_range4)
        # print(attention4.shape)
        long_range4 = attention4 + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)
        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        # outputs = F.dropout(outputs, self.dorp_rate, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        # outputs = F.dropout(outputs, self.dorp_rate, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        # outputs = F.dropout(outputs, self.dorp_rate, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

if __name__ == "__main__":

    device = torch.device('cpu')
    # model = TransBTS(in_channel=4, out_channel=4,training=True).to(device)
    model = TransBTS(4,4,True).to(device)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)

    input = torch.ones([2,4,64,256,256]).to(device)
    output = model(input)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)

