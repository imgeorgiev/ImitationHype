import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from layers import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    LearnablePosEncoding2D,
)
from typing import Optional
from torch.distributions import Normal


class Transformer(nn.Module):
    def __init__(
        self,
        d: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        L: Optional[int] = None,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        default_pos_encoding: bool = True,
    ):
        """Basic transformer implementation with encoder and decoder. Made to be as basic
        as possible but also flexible to be put into ACT.

        Args:
            d: hidden dimension
            h: number of heads
            d_ff: feed forward dimension
            num_layers: number of layers for encoder and decoder
            L: sequence length
            dropout: dropout rate
            src_vocab_size: size of source vocabulary
            tgt_vocab_size: size of target vocabulary
        """
        super(Transformer, self).__init__()
        if src_vocab_size:
            self.src_embed = nn.Embedding(src_vocab_size, d)
        if tgt_vocab_size:
            self.tgt_embed = nn.Embedding(tgt_vocab_size, d)

        self.default_pos_encoding = default_pos_encoding
        assert (
            default_pos_encoding or L == None
        ), "Must provide L if not using default pos encoding"
        if self.default_pos_encoding:
            self.pos_encoding = PositionalEncoding(d, L)

        # Use module list as we need to pass in masks at each step
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d, h, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d, h, d_ff, dropout) for _ in range(num_layers)]
        )

        if tgt_vocab_size:
            self.fc = nn.Linear(d, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # TODO no clue what is happening here
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        L = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, L, L), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, auto_masks=False):
        if auto_masks:
            src_mask, tgt_mask = self.generate_mask(src, tgt)
        else:
            src_mask = tgt_mask = None

        # check if class has been initialized with src_vocab_size
        if hasattr(self, "src_embed"):
            src = self.src_embed(src)
        if hasattr(self, "tgt_embed"):
            tgt = self.tgt_embed(tgt)

        if self.default_pos_encoding:
            src = self.pos_encoding(src)
            tgt = self.pos_encoding(tgt)

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # encoder first
        encoded = src
        for enc_layer in self.encoder_layers:
            encoded = enc_layer(encoded, src_mask)

        # decoder next
        decoded = tgt
        for dec_layer in self.decoder_layers:
            decoded = dec_layer(decoded, encoded, src_mask, tgt_mask)

        if hasattr(self, "fc"):
            decoded = self.fc(decoded)

        return decoded


class StyleEncoder(nn.Module):
    def __init__(
        self,
        act_len: int,
        act_dim: int,
        hidden_dim: int,
        latent_dim: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(StyleEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.act_len = act_len
        self.hidden_dim = hidden_dim
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, h, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.cls_embedding = nn.Parameter(torch.rand(1, hidden_dim))
        self.action_projection = nn.Linear(act_dim, hidden_dim)
        self.qpos_projection = nn.Linear(act_dim, hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim * 2)
        self.pos_encoding = PositionalEncoding(hidden_dim, act_len + 2)

    def forward(self, qpos, actions):
        bsz = qpos.shape[0]
        qpos = self.qpos_projection(qpos)[:, None]
        actions = self.action_projection(actions)
        cls = self.cls_embedding.unsqueeze(1).expand(bsz, -1, -1)
        x = torch.cat([cls, qpos, actions], dim=1)
        assert x.shape == (bsz, self.act_len + 2, self.hidden_dim)
        x = x.transpose(0, 1)

        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, None)

        x = x[0]  # take class output only
        x = self.latent_projection(x)
        mu, logstd = x.chunk(2, dim=-1)
        dist = Normal(mu, logstd.exp())
        return dist


class BCPolicy(torch.nn.Module):
    def __init__(
        self,
        num_img,
        qpos_dim,
        act_len,
        hidden_dim=512,
        hidden_size=[2048, 2048],
        freeze_backbone_bn: bool = True,
    ):
        super(BCPolicy, self).__init__()
        self.num_img = num_img
        self.qpos_dim = qpos_dim
        self.act_len = act_len
        self.hidden_size = hidden_size

        # first make backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # freeze batchnorm layers
        if freeze_backbone_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()  # Set BatchNorm layer to evaluation mode
                    for param in module.parameters():
                        param.requires_grad = False

        # get bottleneck dim
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 480, 640)
            _dummy_output = self.backbone(dummy_input)[0]
            print(f"dummy_output shape: {_dummy_output.shape}")

        # image translation layer to ensure we get to the correct hidden dim
        self.conv = nn.Conv2d(_dummy_output.shape[0], hidden_dim, 1)
        # qpos input processing layer to upsample to hidden_dim
        self.qpos_fc = nn.Linear(qpos_dim, hidden_dim)

        # now make BC model
        self.bottleneck_dim = hidden_dim * _dummy_output.shape[1:].numel()
        hidden_size = [self.num_img * self.bottleneck_dim + hidden_dim] + hidden_size
        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(nn.ELU())
            layers.append(nn.LayerNorm(hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[-1], qpos_dim * act_len))
        self.model = nn.Sequential(*layers)

        print(self.backbone)
        print(self.model)

    def forward(self, img, qpos, actions=None):
        """actions is none to conform with other models"""
        if img.dim() == 5:
            shp = img.shape
            z = self.backbone(img.view(-1, *shp[2:]))
            z = self.conv(z)
            # view is more efficient but only works on shp[0] != 1
            if shp[0] == 1:
                z = z.reshape((shp[0], -1))
            else:
                z = z.view(shp[0], -1)
        elif img.dim() in (3, 4):
            z = self.backbone(img)
            z = self.conv(z)
        else:
            raise ValueError("Invalid input image dimensions")

        qpos = self.qpos_fc(qpos)
        z = torch.cat([z, qpos], dim=1)
        z = self.model(z)
        z = z.reshape(-1, self.act_len, self.qpos_dim)
        return z, None

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ACT(nn.Module):
    """Action Chunking Transformer from Learning Fine-Grained Bimanual
    Manipulation with Low-Cost Hardware. https://arxiv.org/abs/2304.13705
    Heavily modified and reimplemented with different quirks

    Args:
        qpos_dim: dimension of qpos for both inputs and state
        act_len: length of output action sequence
        style_latent_dim: dimension of style latent
        style_encoder: style encoder model digested in directly as torch Module
        hidden_dim: hidden dimension for ACT model
        backbone: backbone model for image processing
        transformer: transformer model for sequence processing
        image_pos_enc_scale: scale for image positional encoding used as img * scale + encoding;
        freeze_backbone_bn: freeze batchnorm layers in backbone; not sure if it does anything honestly
    """

    def __init__(
        self,
        qpos_dim: int,
        act_len: int,
        style_latent_dim: int,
        hidden_dim: int,
        style_encoder: nn.Module,
        backbone: nn.Module,
        transformer: nn.Module,
        image_pos_enc_scale: float = 0.1,
        freeze_backbone_bn: bool = True,
    ):
        assert qpos_dim > 0, "qpos_dim must be greater than 0"
        assert act_len > 0, "act_len must be greater than 0"
        assert hidden_dim > 0, "hidden_dim must be greater than 0"

        super(ACT, self).__init__()
        self.qpos_dim = qpos_dim
        self.act_len = act_len
        self.hidden_dim = hidden_dim
        self.style_latent_dim = style_latent_dim

        # first make backbone and remove final avg pool layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # freeze batchnorm layers
        if freeze_backbone_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()  # Set BatchNorm layer to evaluation mode
                    for param in module.parameters():
                        param.requires_grad = False

        # get bottleneck dim
        with torch.no_grad():
            # use standard resnet input sizes
            dummy_input = torch.randn(1, 3, 640, 480)
            _dummy_output = self.backbone(dummy_input)[0]
            print(f"dummy_output shape: {_dummy_output.shape}")

        self.style_encoder = style_encoder
        self.style_fc = nn.Linear(style_latent_dim, hidden_dim)

        # image translation layer to ensure we get to the correct hidden_dim
        self.conv = nn.Conv2d(_dummy_output.shape[0], hidden_dim, 1)
        # qpos input processing layer to upsample to hidden_dim
        self.qpos_fc = nn.Linear(qpos_dim, hidden_dim)

        # output heads
        self.action_fc = nn.Linear(hidden_dim, qpos_dim)

        # positional embeddings for transformer model
        self.image_embed = LearnablePosEncoding2D(hidden_dim, scale=image_pos_enc_scale)
        self.qpos_pos_enc = nn.Parameter(torch.rand(1, hidden_dim))
        self.style_pos_enc = nn.Parameter(torch.rand(1, hidden_dim))
        self.query_pos_enc = nn.Parameter(torch.rand(act_len, hidden_dim))

        # transformer model
        self.transformer = transformer

    def forward(self, img, qpos, actions=None):
        """
        Arggs:
            img: torch.Tensor (bsz, num_imgs, C, H, W) or (bsz, C, H, W)
                images are fed directly into resnet at original resolution
            qpos: (bsz, qpos_dim) current state of joints
            actions: (bsz, act_len, qpos_dim) if used to learn a style encoding;
                If None, there is no style encoding and instead the latent is sampled
                from a normal distribution (0,1). Used for inference.
        """
        bsz = img.shape[0]

        # Encode images into latent
        if img.dim() == 5:
            shp = img.shape
            num_imgs = img.shape[1]
            img = self.backbone(img.view(-1, *shp[2:]))
            img = self.conv(img)
            img = img.view(bsz, num_imgs, *img.shape[1:-1], -1)
        elif img.dim() in (3, 4):
            img = self.backbone(img)
            img = self.conv(img)
        else:
            raise ValueError("Invalid input image dimensions")

        img = self.image_embed(img)

        # encode style latent
        if actions is None:
            style = torch.zeros((bsz, self.hidden_dim)).to(img.device)
            style_dist = None
        else:
            style_dist = self.style_encoder(qpos, actions)
            style = style_dist.rsample()
            style = self.style_fc(style)
        style = style + self.style_pos_enc
        assert style.shape == (bsz, self.hidden_dim)

        # encode qpos into latent
        qpos = self.qpos_fc(qpos)
        qpos = qpos + self.qpos_pos_enc
        assert qpos.shape == (bsz, self.hidden_dim)

        # concat all latents into transformer input
        z = torch.cat([qpos[None], style[None], img], dim=0)
        assert z.shape[1:] == (bsz, self.hidden_dim)

        target = self.query_pos_enc.unsqueeze(1).expand(-1, bsz, -1)

        z = self.transformer(z, target)
        assert z.shape == (self.act_len, bsz, self.hidden_dim)

        # put in batch-first format and pass through final heads
        z = z.transpose(0, 1)
        a_hat = self.action_fc(z)
        return a_hat, style_dist
