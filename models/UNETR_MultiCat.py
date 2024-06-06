# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
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
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size*4,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size*4,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size*4,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size*4,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
        self.output1 = nn.Sequential(
            nn.Conv3d(4, 4, 1, 1),
            nn.Softmax(dim=1)
        )
        self.output2 = nn.Sequential(
            nn.Conv3d(self.hidden_size*4, 4, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.output3 = nn.Sequential(
            nn.Conv3d(128, 4, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.output4 = nn.Sequential(
            nn.Conv3d(64, 4, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.output5 = nn.Sequential(
            nn.Conv3d(32,4,1,1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size*4)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        # print(x_in[:, 0:1, :].shape)
        x_1, hidden_states_out_1 = self.vit(x_in[:, 0:1, :])
        x_2, hidden_states_out_2 = self.vit(x_in[:, 1:2, :])
        x_3, hidden_states_out_3 = self.vit(x_in[:, 2:3, :])
        x_4, hidden_states_out_4 = self.vit(x_in[:, 3:4, :])
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        # print(x.shape)
        # hidden_states_out = torch.cat([hidden_states_out_1,hidden_states_out_2],dim=1)

        enc1 = self.encoder1(x_in)
        x2 = torch.cat([hidden_states_out_1[3], hidden_states_out_2[3], hidden_states_out_3[3], hidden_states_out_4[3]],
                       dim=1)
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = torch.cat([hidden_states_out_1[6], hidden_states_out_2[6], hidden_states_out_3[6], hidden_states_out_4[6]],
                       dim=1)
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = torch.cat([hidden_states_out_1[9], hidden_states_out_2[9], hidden_states_out_3[9], hidden_states_out_4[9]],
                       dim=1)
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits
        # print(dec4.shape)

if __name__ == "__main__":
    device = torch.device('cpu')
    model = UNETR(
        in_channels=1,
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
    input = torch.ones([2,4,128,240,240]).to(device)
    # num_params = 0
    # for param in model.parameters():
    #     num_params += param.numel()
    # # print(model)
    # print('Total number of parameters: %d' % num_params)
    output = model(input)
    print(output.shape)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)
    # print(output[4].shape)

