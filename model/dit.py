import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat
from jaxtyping import Float, Int, Bool
from x_transformers.x_transformers import RotaryEmbedding

from model.modules import (
    TimestepEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
)

BASE_DIM = 256

# Conditional embedding for f0, rms, cvec

class CondEmbedding(nn.Module):
    def __init__(self, cvec_dim: int, cond_dim: int):
        super().__init__()
        self.cvec_dim = cvec_dim
        self.cond_dim = cond_dim
        self.f0_embed = nn.Linear(1, cond_dim)
        self.rms_embed = nn.Linear(1, cond_dim)
        self.cvec_embed = nn.Linear(cvec_dim, cond_dim)

    def forward(
            self,
            f0: Float[torch.Tensor, "b n"],
            rms: Float[torch.Tensor, "b n"],
            cvec: Float[torch.Tensor, "b n d"]
        ):
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if rms.ndim == 2:
            rms = rms.unsqueeze(-1)

        cond = self.f0_embed(f0 / 1200) + self.rms_embed(rms) + self.cvec_embed(cvec)

        return cond


# noised input audio and context mixing embedding

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.mel_embed = nn.Linear(mel_dim, out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim)

    def forward(self, x: Float[torch.Tensor, "b n d1"], cond_embed: Float[torch.Tensor, "b n d2"]):
        x = self.mel_embed(x)
        x = torch.cat((x, cond_embed), dim = -1)
        x = self.proj(x)
        return x



# backbone using DiT blocks

class DiT(nn.Module):
    def __init__(self,
                 dim: int, depth: int, head_dim: int, dropout: float = 0.1, ff_mult: int = 4,
                 mel_dim: int = 128, num_speaker: int = 1, cvec_dim: int = 768, init_std: float = 0.02, mup_enabled: bool = False):
        super().__init__()

        self.num_speaker = num_speaker
        self.spk_embed = nn.Embedding(num_speaker, dim)
        self.null_spk_embed = nn.Embedding(1, dim)
        self.time_embed = TimestepEmbedding(dim)
        self.cond_embed = CondEmbedding(cvec_dim, dim)
        self.input_embed = InputEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(head_dim)

        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    head_dim = head_dim,
                    ff_mult = ff_mult,
                    dropout = dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.output = nn.Linear(dim, mel_dim)

        self.init_std = init_std
        ## mup
        self.mup_enabled = mup_enabled
        self.mup_multipler = self.dim / BASE_DIM

        self.apply(self._init_weights)
        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.attn_norm.proj.weight, 0)
            torch.nn.init.constant_(block.attn_norm.proj.bias, 0)

            if self.mup_enabled:
                depth_std = self.init_std / math.sqrt(2 * self.depth * self.mup_multipler)
                torch.nn.init.normal_(block.attn.attn_out.weight, mean=0.0, std=depth_std)
                torch.nn.init.normal_(block.mlp.mlp_out.weight, mean=0.0, std=depth_std)

        torch.nn.init.constant_(self.norm_out.proj.weight, 0)
        torch.nn.init.constant_(self.norm_out.proj.bias, 0)
        torch.nn.init.constant_(self.output.weight, 0)
        torch.nn.init.constant_(self.output.bias, 0)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            init_std = self.init_std / self.mup_multipler if self.mup_enabled else self.init_std
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)


    def forward(
        self,
        x: Float[torch.Tensor, "b n d1`"],     # nosied input audio
        spk: Int[torch.Tensor, "b"],         # speaker
        f0: Float[torch.Tensor, "b n"],     # f0
        rms: Float[torch.Tensor, "b n"],    # rms
        cvec: Float[torch.Tensor, "b n d2"],  # cvec
        time: Float[torch.Tensor, "b"] | Float[torch.Tensor, "b n"],  # time step
        drop_spk: bool,  # cfg for speaker
        mask: Bool[torch.Tensor, "b n"] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if drop_spk:
            spk_embed = self.null_spk_embed.weight
        else:
            spk_embed = self.spk_embed(spk)
        # the spk embed is added to the time embed
        t = t + spk_embed
        cond_embed = self.cond_embed(f0, rms, cvec)
        x = self.input_embed(x, cond_embed)
        
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask = mask, rope = rope)

        x = self.norm_out(x, t)
        if self.mup_enabled:
            x = x / self.mup_multipler
        output = self.output(x)

        return output
