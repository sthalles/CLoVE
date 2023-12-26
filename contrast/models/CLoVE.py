
import torch
import torch.nn as nn
from .base import BaseModel


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()
        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = conv1x1(inner_dim, out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim, affine=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        return x


def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


class CLoVE(BaseModel):
    def __init__(self, base_encoder, args):
        super(CLoVE, self).__init__(base_encoder, args)

        # create the encoder
        self.encoder = base_encoder(head_type='early_return')
        self.projector = Proj_Head()

        self.self_attn = nn.MultiheadAttention(
            256, args.clove_attn_heads, batch_first=True)

        # create the encoder_k
        self.encoder_k = base_encoder(head_type='early_return')
        self.projector_k = Proj_Head()

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        for param_b, param_m in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def featprop(self, feat):
        N, C, H, W = feat.shape
        L = H * W
        feat = feat.view(N, C, L).permute(0, 2, 1)  # [N, L, C]
        # Value transformation
        attn_feat, attn_weights = self.self_attn(feat, feat, feat)

        attn_feat = attn_feat.permute(0, 2, 1)  # [N, C, L]
        return attn_feat.view(N, C, H, W), attn_weights

    def forward(self, images, m=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        gl_view = torch.cat(images[:2], dim=0)
        lo_view = torch.cat(images[2:], dim=0)

        # compute query features
        gl_feats = self.encoder(gl_view)  # queries: NxC
        gl_projs = self.projector(gl_feats)
        gl_sa_maps, gl_sa_weights = self.featprop(gl_projs)
        gl_sa_maps = nn.functional.normalize(gl_sa_maps, dim=1)

        lo_feats = self.encoder(lo_view)
        lo_projs = self.projector(lo_feats)
        lo_sa_maps, lo_sa_weights = self.featprop(lo_projs)
        lo_sa_maps = nn.functional.normalize(lo_sa_maps, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if m is not None:
                self._update_momentum_encoder(m)

            gl_feats_ng = self.encoder_k(gl_view)  # keys: NxC
            gl_projs_ng = self.projector_k(gl_feats_ng)
            gl_projs_ng = nn.functional.normalize(gl_projs_ng, dim=1)

        return gl_sa_maps, lo_sa_maps, gl_projs_ng, gl_sa_weights, lo_sa_weights
