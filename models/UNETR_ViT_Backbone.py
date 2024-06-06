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

from timm.models.vision_transformer import PatchEmbed, Block
import random

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
        self.decoder_embed = nn.Linear(768,384,bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1,1,384))
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 2048+1,384),requires_grad=False)
        self.decoder_block = nn.ModuleList([
            Block(384, 16, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(8)])
        self.decoder_norm = nn.LayerNorm(384)
        self.decoder_pred = nn.Linear(384,16*16*16,bias=True)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size*4)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def patchify(self, imgs):
        """
        imgs: (N, 1, S, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        s = imgs.shape[2] // p
        h = w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 4, s, p, h, p, w, p))
        x = torch.einsum('ncsohpwq->nshwopqc', x)
        x = x.reshape(shape=(imgs.shape[0], s * h * w * 4, p**3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *4)
        imgs: (N, 4, H, W)
        """
        p = 16
        s = 4
        h = w = 15
        # assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], s, h, w, p, p, p,4))
        x = torch.einsum('nshwopqc->ncsohpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 4, s*p,h * p, h * p))
        return imgs

    def forward(self, x_in):
        random_num = random.randint(0,3)
        mask_list = [0,1,2,3]
        x = []
        for i in mask_list:
            if i == random_num:
                x.append(self.mask_token.repeat(x_in.shape[0],900,1))
            else:
                x.append(self.decoder_embed(self.vit(x_in[:, i:i+1, :])[0]))
                # print(x[i].shape)
        x = torch.cat([x[0],x[1],x[2],x[3]], dim=1)
        for blk in self.decoder_block:
            x = blk(x)
        x = self.decoder_norm(x)
        output = self.decoder_pred(x)
        # print("output", output.shape)
        mask_x = self.patchify(x_in)
        loss = (output - mask_x) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss.unsqueeze(2) * mask_x).sum() / mask_x.sum()
        output = self.unpatchify(output)
        # print(output.shape)
        if self.training:
            return loss,output
        else:
            return self.unpatchify(output)

class discriminiator(nn.Module):
    def __init__(self,d=128):
        super(discriminiator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4,8,3,1,1),
            nn.PReLU(8),
            nn.Conv3d(8,8,16,16),
            nn.PReLU(8,8)
        )
        self.linear = nn.Sequential(
            nn.Linear(8*4*15*15,512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self,input):
        x = self.net(input)
        # print(x.shape)
        x = x.view(-1,8*4*15*15)
        x = self.linear(x)
        # print(x.shape)
        return x

if __name__ == "__main__":
    device = torch.device('cuda')
    model = UNETR(
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
    input = torch.ones([2,4,64,240,240]).to(device)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    # print(model)
    print('Total number of parameters: %d' % num_params)
    loss, output = model(input)
    print(loss)
    D = discriminiator().to(device)
    output = D(output)
    print(output)
    print(output.shape)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)
    # print(output[4].shape)

