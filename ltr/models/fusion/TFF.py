
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from torch import Tensor
from einops.layers.torch import Rearrange

class Fusion(nn.Module):
    def __init__(self, dim, head, dim_fc=512, dropout=0.1):
        super(Fusion, self).__init__()
        self.rgb_sa_norm1 = nn.LayerNorm(dim)
        self.rgb_sa       = nn.MultiheadAttention(dim, head)
        self.rgb_sa_drop  = nn.Dropout(dropout)
        self.rgb_sa_mlp   = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fc, dim),
            nn.Dropout(dropout))
        self.dvs_sa_norm1 = nn.LayerNorm(dim)
        self.dvs_sa       = nn.MultiheadAttention(dim, head)
        self.dvs_sa_drop  = nn.Dropout(dropout)
        self.dvs_sa_mlp   = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fc, dim),
            nn.Dropout(dropout))
        self.rgb_fa_norm1 = nn.LayerNorm(dim)
        self.rgb_fa       = nn.MultiheadAttention(dim, head)
        self.rgb_fa_drop  = nn.Dropout(dropout)
        self.rgb_fa_mlp   = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fc, dim),
            nn.Dropout(dropout))
        self.dvs_fa_norm1 = nn.LayerNorm(dim)
        self.dvs_fa_drop  = nn.Dropout(dropout)
        self.dvs_fa       = nn.MultiheadAttention(dim, head)
        self.dvs_fa_mlp   = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fc, dim),
            nn.Dropout(dropout))

    def forward(self, rgb: Tensor, dvs: Tensor) -> Tensor:
        ## rgb self-attention
        rgb_ = self.rgb_sa_norm1(rgb)
        rgb  = self.rgb_sa_drop(self.rgb_sa(rgb_, rgb_, rgb_)[0]) + rgb  ## Multi-head Attention
        rgb  = self.rgb_sa_mlp(rgb) + rgb                                ## MLP
        ## dvs self-attention
        dvs_ = self.dvs_sa_norm1(dvs)
        dvs  = self.dvs_sa_drop(self.dvs_sa(dvs_, dvs_, dvs_)[0]) + dvs  ## Multi-head Attention
        dvs = self.dvs_sa_mlp(dvs) + dvs                                 ## MLP
        ## features stack
        fea = torch.cat([rgb, dvs], 1)
        ## rgb fusion-attention
        fea_ = self.rgb_fa_norm1(fea)
        rgb  = self.rgb_fa_drop(self.rgb_fa(rgb, fea_, fea_)[0]) + rgb  ## Multi-head Attention
        rgb  = self.rgb_fa_mlp(rgb) + rgb                               ## MLP
        ## dvs fusion-attention
        fea_ = self.dvs_fa_norm1(fea)
        dvs  = self.dvs_fa_drop(self.dvs_fa(dvs, fea_, fea_)[0]) + dvs  ## Multi-head Attention
        dvs  = self.dvs_fa_mlp(dvs) + dvs                               ## MLP
        return rgb, dvs

class FiT(nn.Module):
    def __init__(self, num_patches: int, patch_dim: int, head: int=2, num_fusion_layer: int=2):
        super(FiT, self).__init__()
        self.num_patches      = num_patches
        self.patch_dim        = patch_dim
        self.head             = head
        self.num_fusion_layer = num_fusion_layer
        self._embedding()
        self.fusion_l         = nn.ModuleList([Fusion(dim=int(self.patch_dim/2), head=self.head) for _ in range(self.num_fusion_layer)])
        self.fusion_h         = nn.ModuleList([Fusion(dim=self.patch_dim, head=self.head) for _ in range(self.num_fusion_layer)])
        self._decoder()

    def _embedding(self):
        self.embedding_rgb_l = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
                    nn.LayerNorm(128*4*4),
                    nn.Linear(128*4*4, int(self.patch_dim/2)),
                    nn.LayerNorm(int(self.patch_dim/2)),
                    nn.Dropout(0.1))
        self.embedding_dvs_l = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
                    nn.LayerNorm(128*4*4),
                    nn.Linear(128*4*4, int(self.patch_dim/2)),
                    nn.LayerNorm(int(self.patch_dim/2)),
                    nn.Dropout(0.1))
        self.embedding_rgb_h = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 2, p2 = 2),
                    nn.LayerNorm(256*2*2),
                    nn.Linear(256*2*2, self.patch_dim),
                    nn.LayerNorm(self.patch_dim),
                    nn.Dropout(0.1))
        self.embedding_dvs_h = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 2, p2 = 2),
                    nn.LayerNorm(256*2*2),
                    nn.Linear(256*2*2, self.patch_dim),
                    nn.LayerNorm(self.patch_dim),
                    nn.Dropout(0.1))

    def _decoder(self):
        self.decoder_l = nn.Sequential(
                    nn.LayerNorm(128),
                    Rearrange('b t d -> b d t'),
                    nn.Linear(162, 36*36),
                    nn.LayerNorm(36*36),
                    nn.Dropout(0.1),
                    Rearrange('b c (w h) -> b c w h', h=36, w=36))
        self.decoder_h = nn.Sequential(
                    nn.LayerNorm(256),
                    Rearrange('b t d -> b d t'),
                    nn.Linear(162, 18*18),
                    nn.LayerNorm(18*18),
                    nn.Dropout(0.1),
                    Rearrange('b c (w h) -> b c w h', h=18, w=18))

    def forward(self, rgb_l, dvs_l, rgb_h, dvs_h):
        rgb_l = self.embedding_rgb_l(rgb_l)
        dvs_l = self.embedding_dvs_l(dvs_l)
        rgb_h = self.embedding_rgb_h(rgb_h)
        dvs_h = self.embedding_dvs_h(dvs_h)
        for layer in self.fusion_l:
            rgb_l, dvs_l = layer(rgb_l, dvs_l)
        for layer in self.fusion_h:
            rgb_h, dvs_h = layer(rgb_h, dvs_h)
        feature_l = torch.cat([rgb_l, dvs_l], 1)
        feature_h = torch.cat([rgb_h, dvs_h], 1)
        feature_l = self.decoder_l(feature_l)
        feature_h = self.decoder_h(feature_h)
        return feature_l, feature_h
