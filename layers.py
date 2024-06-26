import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        """Multi-head attention layer. Basic building block of transformer
        d: input dimension
        h: number of heads
        dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        assert d % h == 0, "d must be divisible by h"

        self.d = d  # input dimension
        self.d_k = d // h  # dimension of each head
        self.h = h

        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)

    def split_heads(self, x):
        bsz, L, d = x.size()
        assert d == self.d
        return x.view(bsz, L, self.h, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        bsz, h, L, d_k = x.size()
        assert h == self.h
        assert d_k == self.d_k
        return x.transpose(1, 2).contiguous().view(bsz, L, self.d)

    def attention(self, q, k, v, mask=None):
        """
        Attention function
        q: query
        k: key
        v: value
        attn = softmax((Q K^T) / sqrt(d_k)) V
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        dist = F.softmax(scores, dim=-1)
        return torch.matmul(dist, v)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        output = self.attention(q, k, v, mask)
        return self.W_o(self.combine_heads(output))


class PositionalEncoding(nn.Module):
    """Standard positional encoding for text tasks as used in the orignal transformer paper

    Args:
        d: hidden dimension for embeddings
        L: maximum sequence length
    """

    def __init__(self, d: int, L: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(L, d)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer to ensure that this modele isn't trained
        self.register_buffer("pe", pe[None])

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PositionalEncoding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Copied from DETR repo; can be heavily optimized

    Args:
        num_pos_feats: hidden dimesion (I think)
        temperature: ???
        normalize: whether to normalize embeddings
        scale: used for normalization
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super(PositionalEncoding2D, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x must be (bsz, ..., hidden_dim, H, W)
        not_mask = torch.ones(x.shape[-2:])[None].to(x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return x + pos


class LearnablePosEncoding2D(nn.Module):
    """Learnable positional encoding for 2D inputs; inspured by DETR paper"""

    def __init__(
        self,
        hidden_dim: int,
        num_pos_feats: int = 64,
        scale: float = 1.0,
    ):
        assert hidden_dim % 2 == 0, "Hidden dimension must be even"
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.row_embed = nn.Parameter(torch.rand(num_pos_feats, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(num_pos_feats, hidden_dim // 2))

    def forward(self, x):
        H, W = x.shape[-2:]
        col = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)
        row = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        pos = torch.cat([col, row], dim=-1).flatten(0, 1).unsqueeze(1)
        if x.dim() == 5:
            num_imgs = x.shape[1]
            pos = pos.repeat(num_imgs, 1, 1)
            x = x.transpose(1, 2)
        x = x.flatten(2).permute(2, 0, 1)
        return self.scale * x + pos


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        """
        Args:
            d: input dimension
            h: number of heads
            d_ff: feed forward dimension
            dropout: dropout rate
        """
        assert dropout >= 0.0 and dropout <= 1.0, "dropout must be in [0.0, 1.0]"
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d, h)
        self.linear1 = nn.Linear(d, d_ff)
        self.linear2 = nn.Linear(d_ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        """
        Args:
            d: input dimension
            h: number of heads
            d_ff: feed forward dimension
            dropout: dropout rate
        """
        assert dropout >= 0.0 and dropout <= 1.0, "dropout must be in [0.0, 1.0]"
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d, h)
        self.cross_attn = MultiHeadAttention(d, h)
        self.linear1 = nn.Linear(d, d_ff)
        self.linear2 = nn.Linear(d_ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, src_mask, target_mask):
        attn = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attn))
        attn = self.cross_attn(x, cross, cross, src_mask)
        x = self.norm2(x + self.dropout(attn))
        ff = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm3(x + self.dropout(ff))
        return x
