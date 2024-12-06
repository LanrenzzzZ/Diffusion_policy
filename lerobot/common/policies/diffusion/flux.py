from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lerobot.common.policies.diffusion.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)



@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    
import math

class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self,  
            input_dim=14,
            output_dim=14,
            n_head=12,
            n_emb=512):
        super().__init__()

        self.in_channels = input_dim
        self.out_channels = output_dim
        
        self.hidden_size = n_emb
        self.num_heads = n_head
        self.action_in = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.GELU(approximate="tanh"),
            nn.Linear(512, 512)
        )
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                )
                for _ in range(6)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=4.0)
                for _ in range(10)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        action: Tensor,
        timesteps: Tensor,
        cond: Tensor,
    ) -> Tensor:
       
        action = self.action_in(action)                         #  [b, 16, 512]
        vec = self.time_in(timestep_embedding(timesteps, 256))  #  [b, 1, 512]

        for block in self.double_blocks:
            img, cond = block(img=action, txt=cond, vec=vec)

        img = torch.cat((cond, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec)
        img = img[:, cond.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
