
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import random
from pytorch_wavelets import DWTForward, DWTInverse
from typing import Tuple, Optional

"""
Please check:
1. class DINO_AugSeg(nn.Module): build the DINO-AugSeg
2. class AttentionCrossDecoder_WT_ALL(nn.Module): include CG-Fuse and WT-Aug of the paper
    1> CG-Fuse: contextual-guided feature fusion module to leverage the high-level contextual information from DINOv3 feature
    2> WT-Aug (only used in training): wavelet-based feature-level augmentation method for DINOv3 features
3. The paper could be download in : https://www.arxiv.org/abs/2601.08078
"""

class ImageAugmentor(nn.Module):
    def __init__(self, prob=0.7, drop_rate=0.1):
        super().__init__()
        self.prob = prob
        self.drop_rate = drop_rate

    # ----------- augmentation ops ----------------
    def aug_brightness(self, x):
        """Random brightness scaling."""
        scale = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
        return x * scale

    def aug_motion(self, x, kernel=7):
        """Horizontal motion blur."""
        b, c, h, w = x.shape
        k = torch.ones((c, 1, 1, kernel), device=x.device) / kernel
        return F.conv2d(x, k, padding=(0, kernel // 2), groups=c)

    def aug_poisson(self, x):
        """Poisson noise."""
        x_pos = torch.clamp(x, min=0)
        return torch.poisson(x_pos)

    def aug_random_zero(self, x):
        """Randomly mask pixels."""
        mask = (torch.rand_like(x) > self.drop_rate).float()
        return x * mask

    # ----------- forward ----------------
    def forward(self, x):  # x: B×C×H×W
        if random.random() > self.prob:
            return x
        # if not self.training:
        #     return x

        aug_type = random.choice(["brightness", "motion", "poisson", "random_zero"])

        if aug_type == "brightness":
            return self.aug_brightness(x)
        elif aug_type == "motion":
            return self.aug_motion(x)
        elif aug_type == "poisson":
            return self.aug_poisson(x)
        elif aug_type == "random_zero":
            return self.aug_random_zero(x)

        return x


## feature augmentation module
class FeatureAugmentor(nn.Module):
    def __init__(self, prob=0.7, drop_rate=0.2):
        super().__init__()
        self.prob = prob
        self.drop_rate = drop_rate

    def forward(self, feats):
        f_list = list(feats)

        if random.random() > self.prob:
            return feats

        # randomly choose one feature level to augment
        idx = random.randint(0, len(f_list) - 1)
        feat = f_list[idx]

        # randomly choose augmentation method
        aug_type = random.choice(["brightness", "motion", "poisson", "random_zero"])

        if aug_type == "brightness":
            feat = self.brightness_aug(feat)

        elif aug_type == "motion":
            feat = self.motion_blur(feat)

        elif aug_type == "poisson":
            feat = self.poisson_noise(feat)

        elif aug_type == "random_zero":
            feat = self.random_zero_mask(feat)

        f_list[idx] = feat
        return tuple(f_list)

    # -------------------------------------
    # Brightness
    # -------------------------------------
    def brightness_aug(self, feat):
        alpha = random.uniform(0.7, 1.3)
        return feat * alpha

    # -------------------------------------
    # Motion Blur
    # -------------------------------------
    def motion_blur(self, feat):
        kernel_size = 7
        kernel = torch.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0
        kernel /= kernel.sum()

        kernel = kernel.unsqueeze(0).unsqueeze(0).to(feat.device)
        B, C, H, W = feat.shape
        kernel = kernel.repeat(C, 1, 1, 1)

        return F.conv2d(feat, kernel, padding=kernel_size//2, groups=C)

    # -------------------------------------
    # Poisson Noise
    # -------------------------------------
    def poisson_noise(self, feat):
        min_val = feat.min()
        shifted = feat - min_val   # ensure >= 0

        scale = 20
        shifted = shifted * scale

        # torch.poisson only runs on CPU
        noisy = torch.poisson(shifted.cpu()).to(feat.device)

        noisy = noisy / scale
        noisy = noisy + min_val

        return noisy

    # -------------------------------------
    # Random Zero Masking (Dropout-style)
    # -------------------------------------
    def random_zero_mask(self, feat):
        """
        Randomly sets positions to zero, similar to your example:
        mask = (rand > drop_rate)
        """
        mask = (torch.rand_like(feat) > self.drop_rate).float()
        return feat * mask


# -----------------------
# Helpers: window partition / reverse (Swin-style)
# -----------------------
def window_partition(x: torch.Tensor, window_size: Tuple[int,int]) -> torch.Tensor:
    """
    x: (B, H, W, C)
    return: (num_windows*B, Wh*Ww, C)
    """
    B, H, W, C = x.shape
    Wh, Ww = window_size
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    windows = x.view(-1, Wh * Ww, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: Tuple[int,int], H: int, W: int) -> torch.Tensor:
    """
    windows: (num_windows*B, Wh*Ww, C)
    return: (B, H, W, C)
    """
    Wh, Ww = window_size
    B = int(windows.shape[0] // (H // Wh * W // Ww))
    x = windows.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# -----------------------
# RoPE utilities
# -----------------------
def build_rope_cache(seq_len: int, dim_head: int, base: int = 10000, device=None) -> torch.Tensor:
    """Return (1, seq_len, dim_head) containing [sin, cos] concat (sin first then cos),
       encoded as (seq_len, dim_head) where dim_head is even and output has shape (1, seq_len, dim_head).
       We'll pack as [sin, cos] interleaved when applying.
    """
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2, device=device).float() / dim_head))
    sinusoid = torch.einsum("i,j->ij", pos, inv_freq)  # (seq_len, dim_head/2)
    sin = sinusoid.sin()
    cos = sinusoid.cos()
    # Interleave sin and cos into dim_head: [sin0, cos0, sin1, cos1, ...]
    sin_cos = torch.stack([sin, cos], dim=-1).reshape(seq_len, dim_head)
    return sin_cos.unsqueeze(0)  # (1, seq_len, dim_head)

def apply_rope_tensor(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    x: (B, nH, N, d) where d = dim_head
    rope_cache: (1, N, d) where d is even and is arranged [sin0,cos0,sin1,cos1,...]
    Implementation uses rotation: for each pair (x_2i, x_2i+1):
      [x_2i', x_2i+1'] = [ x_2i * cos - x_2i+1 * sin, x_2i * sin + x_2i+1 * cos ]
    """
    # x and rope_cache broadcast over batch and heads
    # reshape last dim to (d/2, 2) to vectorize
    B, nH, N, d = x.shape
    assert d % 2 == 0, "head dim must be even for RoPE"
    x_ = x.view(B, nH, N, d // 2, 2)     # (B, nH, N, d/2, 2)
    rope = rope_cache.view(1, N, d // 2, 2).to(x.device)  # (1, N, d/2, 2)
    sin = rope[..., 0]  # (1, N, d/2)
    cos = rope[..., 1]  # (1, N, d/2)
    sin = sin.unsqueeze(0).unsqueeze(1)  # (1,1,N,d/2)
    cos = cos.unsqueeze(0).unsqueeze(1)
    x0 = x_[..., 0]  # (B,nH,N,d/2)
    x1 = x_[..., 1]
    xr0 = x0 * cos - x1 * sin
    xr1 = x0 * sin + x1 * cos
    xr = torch.stack([xr0, xr1], dim=-1).view(B, nH, N, d)
    return xr

# -----------------------
# Relative position bias for windows (Swin-style)
# -----------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size: Tuple[int,int], num_heads: int):
        super().__init__()
        Wh, Ww = window_size
        self.window_size = window_size
        self.num_heads = num_heads
        # number of relative positions
        table_size = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # create index
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[..., 0] += Wh - 1
        relative_coords[..., 1] += Ww - 1
        relative_coords[..., 0] *= 2 * Ww - 1
        relative_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_index", relative_index)

    def forward(self) -> torch.Tensor:
        # returns (num_heads, Wh*Ww, Wh*Ww)
        bias = self.relative_position_bias_table[self.relative_index.view(-1)].view(
            self.relative_index.shape[0], self.relative_index.shape[1], -1
        )  # Wh*Ww, Wh*Ww, num_heads
        bias = bias.permute(2, 0, 1).contiguous()
        return bias  # (num_heads, N, N)

# -----------------------
# Cross Attention Block (global / window / pooled_kv) with RoPE + optional pre-norm
# -----------------------
class CrossAttentionBlock_RoPE(nn.Module):
    def __init__(self,
                 dim_q: int,
                 dim_kv: int,
                 num_heads: int = 8,
                 attn_type: str = "global",   # "global" | "window" | "pooled_kv"
                 window_size: Tuple[int,int] = (7,7),
                 pool_kv: bool = False,
                 pool_stride: int = 2,
                 use_rope: bool = True,
                 pre_norm: bool = True,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 use_rel_pos_bias: bool = True):
        super().__init__()
        assert dim_q % num_heads == 0, "dim_q must be divisible by num_heads"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.window_size = window_size
        self.pool_kv = pool_kv or (attn_type == "pooled_kv")
        self.pool_stride = pool_stride
        self.use_rope = use_rope
        self.pre_norm = pre_norm
        self.use_rel_pos_bias = use_rel_pos_bias and attn_type == "window"

        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        # projections operate on last dim (feature dim)
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_dropout)

        if pre_norm:
            self.norm_q = nn.LayerNorm(dim_q)
            self.norm_kv = nn.LayerNorm(dim_kv)
        else:
            self.post_norm = nn.LayerNorm(dim_q)

        # window relative bias
        if self.use_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.rel_pos_bias = None

        # RoPE cache
        if self.use_rope:
            self._rope_cache: Optional[torch.Tensor] = None
            self._rope_len = 0
            self.head_dim = head_dim

        # pooling
        if self.pool_kv:
            self.pool = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)

    def _maybe_pool_kv(self, kv: torch.Tensor) -> torch.Tensor:
        if self.pool_kv:
            return self.pool(kv)
        return kv

    def _apply_rope_to_QK(self, Q: torch.Tensor, K: torch.Tensor, Nq: int, Nk: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q, K shapes: (B, nH, N, d)
        max_len = max(Nq, Nk)
        if self._rope_cache is None or self._rope_len < max_len or self._rope_cache.device != device:
            self._rope_cache = build_rope_cache(max_len, self.head_dim, device=device)  # (1, max_len, d)
            self._rope_len = max_len
        rope = self._rope_cache  # (1, max_len, d)
        # apply appropriate slices
        Q = apply_rope_tensor(Q, rope[:, :Nq, :].to(device))
        K = apply_rope_tensor(K, rope[:, :Nk, :].to(device))
        return Q, K

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q: (B, Cq, Hq, Wq)
        kv: (B, Ck, Hk, Wk)
        returns: (B, Cq, Hq, Wq) after cross-attention
        """
        B, Cq, Hq, Wq = q.shape
        Bk, Ck, Hk, Wk = kv.shape
        assert B == Bk, "batch size mismatch"

        # optionally pool kv spatially to reduce compute
        kv_proc = self._maybe_pool_kv(kv)  # shape may change

        # flatten to sequences
        q_seq = q.flatten(2).transpose(1,2)          # (B, Nq, Cq)
        kv_seq = kv_proc.flatten(2).transpose(1,2)   # (B, Nk, Ck)
        Nq = q_seq.shape[1]
        Nk = kv_seq.shape[1]

        # pre-norm
        if self.pre_norm:
            q_seq = self.norm_q(q_seq)
            kv_seq = self.norm_kv(kv_seq)

        # project
        Q = self.q_proj(q_seq)   # (B, Nq, Cq)
        K = self.k_proj(kv_seq)  # (B, Nk, Cq)
        V = self.v_proj(kv_seq)  # (B, Nk, Cq)

        # reshape to (B, nH, N, d)
        d = Cq // self.num_heads
        Q = Q.view(B, Nq, self.num_heads, d).permute(0,2,1,3)  # (B, nH, Nq, d)
        K = K.view(B, Nk, self.num_heads, d).permute(0,2,1,3)  # (B, nH, Nk, d)
        V = V.view(B, Nk, self.num_heads, d).permute(0,2,1,3)  # (B, nH, Nk, d)

        # optionally apply RoPE
        if self.use_rope:
            Q, K = self._apply_rope_to_QK(Q, K, Nq, Nk, q.device)

        # If windowed attention, partition into windows (requires Hq,Wq multiple of window size)
        if self.attn_type == "window":
            Wh, Ww = self.window_size
            assert Hq % Wh == 0 and Wq % Ww == 0, "Hq and Wq must be divisible by window size for windowed attention"
            # convert Q,K,V from (B,nH,N,d) -> (B, nH, Hq, Wq, d) -> partition windows -> (num_windows*B, nH, Wh*Ww, d)
            # easier: reshape Q,K,V to (B, nH, Hq, Wq, d)
            Q_hw = Q.permute(0,1,2,3).contiguous().view(B, self.num_heads, Hq, Wq, d).permute(0,2,3,1,4).contiguous()  # (B,Hq,Wq,nH,d)
            K_hw = K.permute(0,1,2,3).contiguous().view(B, self.num_heads, Hq, Wq, d).permute(0,2,3,1,4).contiguous()
            V_hw = V.permute(0,1,2,3).contiguous().view(B, self.num_heads, Hq, Wq, d).permute(0,2,3,1,4).contiguous()

            # Now for each head, partition windows and compute attention per window independently
            # We'll merge head and batch dims for partition
            # Q_windows: (num_windows*B*nH, Wh*Ww, d)
            Q_windows = []
            K_windows = []
            V_windows = []
            for h_idx in range(self.num_heads):
                Q_h = Q_hw[..., h_idx, :].contiguous()  # (B,Hq,Wq,d)
                K_h = K_hw[..., h_idx, :].contiguous()
                V_h = V_hw[..., h_idx, :].contiguous()
                Qw = window_partition(Q_h.permute(0,2,1,3).contiguous(), (Wh, Ww))  # trick: windows assume (B,H,W,C), use permute to match
                Kw = window_partition(K_h.permute(0,2,1,3).contiguous(), (Wh, Ww))
                Vw = window_partition(V_h.permute(0,2,1,3).contiguous(), (Wh, Ww))
                Q_windows.append(Qw)  # list of (num_windows*B, Wh*Ww, d)
                K_windows.append(Kw)
                V_windows.append(Vw)
            # Stack heads along batch axis: shape -> (nH*(num_windows*B), Wh*Ww, d)
            Qw_cat = torch.cat(Q_windows, dim=0)
            Kw_cat = torch.cat(K_windows, dim=0)
            Vw_cat = torch.cat(V_windows, dim=0)

            # reshape for attention: treat concatenated heads as separate batches
            # compute attention
            q_ = Qw_cat.unsqueeze(2)  # (B', L, 1, d) not necessary
            attn = (Qw_cat @ Kw_cat.transpose(-2,-1)) * self.scale  # (B', L, L)

            # add relative pos bias if available
            if self.rel_pos_bias is not None:
                # rel_pos_bias: (nH, L, L). We need to tile to match concatenated heads ordering
                bias = self.rel_pos_bias()  # (nH, L, L)
                # repeat for batch: each head group corresponds to many windows. We tile bias along batch of windows.
                num_windows_per_image = (Hq // Wh) * (Wq // Ww)
                bias_rep = bias.repeat_interleave(repeats=B * num_windows_per_image, dim=0)  # (nH * num_windows*B, L, L)
                attn = attn + bias_rep.to(attn.device)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out_windows = attn @ Vw_cat  # (B', L, d)

            # Now split and merge back per head
            # First, split by heads
            per_head_windows = torch.split(out_windows, out_windows.shape[0] // self.num_heads, dim=0)
            # reconstruct per-head images and then concatenate heads
            head_outputs = []
            for h_idx in range(self.num_heads):
                out_h = per_head_windows[h_idx]  # (num_windows*B, L, d)
                # reverse windows -> (B, Hq, Wq, d)
                out_h_img = window_reverse(out_h, (Wh, Ww), Hq, Wq)  # (B, Hq, Wq, d)
                head_outputs.append(out_h_img)  # list of (B,Hq,Wq,d)
            # stack heads -> (B,Hq,Wq,nH,d) then permute to (B,nH,N,d)
            stacked = torch.stack(head_outputs, dim=3)  # (B,Hq,Wq,nH,d)
            stacked = stacked.view(B, Hq*Wq, self.num_heads, d).permute(0,2,1,3).contiguous()  # (B,nH,N,d)
            # Now merge heads: (B,N,nH,d) -> (B,N,nH*d)
            out = stacked.permute(0,2,1,3).contiguous().view(B, Nq, self.num_heads * d)
            out = self.out_proj(out)
            out = self.proj_drop(out)

            if self.pre_norm:
                # residual q_seq was normalized earlier; add residual and return
                out = out + q_seq
                out = out  # if you want, apply further post-norm variant outside
            if not self.pre_norm:
                out = out + q_seq
                out = self.post_norm(out)

            out = out.transpose(1,2).reshape(B, Cq, Hq, Wq)
            return out

        # ---- Global / pooled_kv attention path ----
        # attention on sequences Q (B,nH,Nq,d), K (B,nH,Nk,d)
        attn = (Q @ K.transpose(-2,-1)) * self.scale  # (B,nH,Nq,Nk)

        # No relative pos bias for global by default (could be added)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).permute(0,2,1,3).contiguous().view(B, Nq, self.num_heads * d)  # (B,Nq,Cq)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # Residual & post-norm
        if self.pre_norm:
            out = out + q_seq
        else:
            out = out + q_seq
            out = self.post_norm(out)

        out = out.transpose(1,2).reshape(B, Cq, Hq, Wq)
        return out

# -------------------------
# helper conv block
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Attention Gate (UNet attention)
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# -------------------------
# wavelet-feature augmentation +  attunet-decoder (lightweight MLP fusion)  
# -------------------------
class WaveletRandomMask(nn.Module):
    def __init__(self, wave="haar", drop_rate=0.3, separate_channels=False):
        super().__init__()
        self.drop_rate = drop_rate
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        self.separate_channels = separate_channels

    def forward(self, x):
        # Only apply in training mode
        if not self.training or self.drop_rate <= 0:
            return x
        
        Yl, Yh = self.dwt(x)
        Yh = Yh[0] # Bx3xCxHxW
        # Random masks
        mask_lp = (torch.rand_like(Yl) > self.drop_rate).float()
        Yl = Yl * mask_lp
        if self.separate_channels:
            # Yh: high-frequency components [B, C, 3, H/2, W/2] (for J=1).
            # Mask each of the 3 subbands separately
            for i in range(3):
                mask_hp = (torch.rand_like(Yh[:, :, i]) > self.drop_rate).float()
                Yh[:, :, i] = Yh[:, :, i] * mask_hp
        else:
            # Mask all high-frequency components together
            mask_hp = (torch.rand_like(Yh) > self.drop_rate).float()
            Yh = Yh * mask_hp

        out = self.idwt((Yl, [Yh]))
        return out


# -------------------------
# att-unet + high feature guided for feature fusion
# -------------------------
# -------------------------
# Cross-attention bottle decoder (lightweight MLP fusion)
# -------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, 
                 attn_type="global",  # "global" or "window"
                 window_size=(7, 7),
                 pre_norm=True,
                 use_residual=True,
                 attn_drop=0.1, proj_drop=0.1,
                 use_rel_pos_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.attn_type = attn_type
        self.window_size = window_size
        self.pre_norm = pre_norm
        self.use_rel_pos_bias = use_rel_pos_bias

        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        if pre_norm:
            self.norm_q = nn.LayerNorm(dim_q)
            self.norm_kv = nn.LayerNorm(dim_kv)
        else:
            self.norm = nn.LayerNorm(dim_q)

        if use_rel_pos_bias and attn_type == "window":
            self.rel_pos_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.rel_pos_bias = None

    def forward_window(self, q, kv, H, W):
        """ Windowed attention (Swin-style). """
        B, Nq, Cq = q.shape
        window_h, window_w = self.window_size
        assert H % window_h == 0 and W % window_w == 0, "Feature map must be divisible by window size"

        # reshape into windows
        q = q.view(B, H, W, Cq)
        kv = kv.view(B, H, W, Cq)

        # partition windows
        q_windows = q.unfold(1, window_h, window_h).unfold(2, window_w, window_w)
        kv_windows = kv.unfold(1, window_h, window_h).unfold(2, window_w, window_w)

        # reshape to (num_windows*B, Wh*Ww, C)
        q_windows = q_windows.contiguous().view(-1, window_h * window_w, Cq)
        kv_windows = kv_windows.contiguous().view(-1, window_h * window_w, Cq)

        return q_windows, kv_windows

    def forward(self, q, kv):
        B, Cq, Hq, Wq = q.shape
        B, Ck, Hk, Wk = kv.shape

        # Flatten
        q = q.flatten(2).transpose(1, 2)   # (B, Hq*Wq, Cq)
        kv = kv.flatten(2).transpose(1, 2) # (B, Hk*Wk, Ck)

        if self.pre_norm:
            q = self.norm_q(q)
            kv = self.norm_kv(kv)

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        if self.attn_type == "window":
            # reshape into windows
            Q, K = self.forward_window(Q, K, Hq, Wq)
            _, V = self.forward_window(Q, V, Hq, Wq)

        # Multi-head split
        Bq = Q.size(0)  # might be B*num_windows
        Q = Q.reshape(Bq, -1, self.num_heads, Cq // self.num_heads).transpose(1, 2)
        K = K.reshape(Bq, -1, self.num_heads, Cq // self.num_heads).transpose(1, 2)
        V = V.reshape(Bq, -1, self.num_heads, Cq // self.num_heads).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale

        # add relative positional bias if available
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias()  # (nH, Wh*Ww, Wh*Ww)
            attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(Bq, -1, Cq)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        if self.attn_type == "window":
            # merge windows back
            out = out.view(B, Hq, Wq, Cq).permute(0, 3, 1, 2).contiguous()
        else:
            out = out.transpose(1, 2).reshape(B, Cq, Hq, Wq)

        if self.use_residual:
            if self.pre_norm:
                out = out + q.transpose(1, 2).reshape(B, Cq, Hq, Wq)
            else:
                out = out + q.transpose(1, 2).reshape(B, Cq, Hq, Wq)
                out = self.norm(out.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, Cq, Hq, Wq)

        return out


# === Decoder with Cross-Attention and WT-augmentation ===
class AttentionCrossDecoder_WT_ALL(nn.Module):
    def __init__(self, enc_channels, final_channels=64, drop_rate=0.3, separate_channels=True, aug_all=True, aug_feat=False):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.aug_all = aug_all
        self.aug_feat = aug_feat
        if self.aug_feat: # do feature level augmentation:  spatial dimension
            self.augmentor = FeatureAugmentor(prob=0.7) 

        # do feature level augmentation:  wavelet dimension
        self.wavelet_mask = WaveletRandomMask(drop_rate=drop_rate, separate_channels=separate_channels)

        # Decoder pathway
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        # self.att4 = AttentionGate(F_g=c3, F_l=c3, F_int=c3 // 2)
        self.att4 = CrossAttentionBlock_RoPE(dim_q=c3, dim_kv=c3, num_heads=4, attn_type="global", 
                                             pool_kv=True, pool_stride=2)  # replaced here
        self.conv4 = nn.Sequential(
            ConvBNReLU(c3 + c3, c3),
            ConvBNReLU(c3, c3)
        )
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        # self.att3 = AttentionGate(F_g=c2, F_l=c2, F_int=c2 // 2)
        self.att3 = CrossAttentionBlock_RoPE(dim_q=c2, dim_kv=c2, num_heads=4, attn_type="global", 
                                             pool_kv=True, pool_stride=2)  # replaced here
        self.conv3 = nn.Sequential(
            ConvBNReLU(c2 + c2, c2),
            ConvBNReLU(c2, c2)
        )

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        # self.att2 = AttentionGate(F_g=c1, F_l=c1, F_int=c1 // 2)
        self.att2 = CrossAttentionBlock_RoPE(dim_q=c1, dim_kv=c1, num_heads=4, attn_type="global", 
                                             pool_kv=True, pool_stride=4)  # replaced here
        self.conv2 = nn.Sequential(
            ConvBNReLU(c1 + c1, c1),
            ConvBNReLU(c1, c1)
        )

        self.up1 = nn.ConvTranspose2d(c1, final_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            ConvBNReLU(final_channels, final_channels),
            ConvBNReLU(final_channels, final_channels)
        )

    def forward(self, feats):
        f1, f2, f3, f4 = feats

        if self.aug_all: ## wavelet augmentation on all features
            # Probability check: only apply augmentation if random > threshold
            prob = random.random()
            if prob <= 0.7:
                # Apply masking to all feature maps
                f1 = self.wavelet_mask(f1)
                f2 = self.wavelet_mask(f2)
                f3 = self.wavelet_mask(f3)
                f4 = self.wavelet_mask(f4)

        if self.aug_feat: ## feature augmentation on one random feature
            f1, f2, f3, f4 = self.augmentor((f1, f2, f3, f4))

        # Decoder pathway
        d4 = self.up4(f4)
        if d4.shape[-2:] != f3.shape[-2:]:
            d4 = F.interpolate(d4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3_att = self.att4(d4, f3)
        # d4 = torch.cat([d4, f3_att], dim=1)
        d4 = torch.cat([f3, f3_att], dim=1)
        d4 = self.conv4(d4)

        d3 = self.up3(d4)
        if d3.shape[-2:] != f2.shape[-2:]:
            d3 = F.interpolate(d3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f2_att = self.att3(d3, f2)
        # d3 = torch.cat([d3, f2_att], dim=1)
        d3 = torch.cat([f2, f2_att], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        if d2.shape[-2:] != f1.shape[-2:]:
            d2 = F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f1_att = self.att2(d2, f1)
        # d2 = torch.cat([d2, f1_att], dim=1)
        d2 = torch.cat([f1, f1_att], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        d1 = self.conv1(d1)
        return d1
 
# -------------------------
# Full ConvNeXtUNet_V2 with decoder selector
# -------------------------
class DINO_AugSeg(nn.Module):
    def __init__(self, encoder, num_classes=1, model_type="tiny", decoder_type="cross_guide_wt_unet", use_wt_aug=True, aug_feat=False):
        """
        encoder: pretrained encoder instance exposing `downsample_layers` and `stages`
        model_type: 'tiny'|'small'|'base'|'large' -> sets enc_channels
        decoder_type: 'attention_unet'|'segformer'|'deeplabv3plus'
        """
        super().__init__()
        self.encoder = encoder
        self.use_wt_aug = use_wt_aug

        # by default freeze encoder weights; user can unfreeze later
        for p in self.encoder.parameters():
            p.requires_grad = False

        if model_type in ("tiny", "small"):
            self.enc_channels = [96, 192, 384, 768]
        elif model_type == "base":
            self.enc_channels = [128, 256, 512, 1024]
        elif model_type == "large":
            self.enc_channels = [192, 384, 768, 1536]
        else:
            raise ValueError("unknown model_type")

        # instantiate decoder
        decoder_type = decoder_type.lower()  
        # CG-Fuse + WT-Aug
        # CG-Fuse: do cross attention between encoder features and decoder features
        # WT-Aug: feature level augmenation on wavelet dimension, only used in training
        if decoder_type == "cross_guide_wt_unet": 
            self.decoder = AttentionCrossDecoder_WT_ALL(self.enc_channels, final_channels=64, aug_all=use_wt_aug, aug_feat=aug_feat)
            decoder_out_channels = 64 
        else:
            raise ValueError("decoder_type must be one of 'attention_unet','segformer','deeplabv3plus'")

        self.out_conv = nn.Conv2d(decoder_out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # ---- encoder forward (same logic as your original) ----
        feats = []
        out = x
        # 1. Encoder: get features from dinov3
        for i, down in enumerate(self.encoder.downsample_layers):
            out = down(out)
            out = self.encoder.stages[i](out)
            feats.append(out)
        # feats: [f1, f2, f3, f4] (shallow -> deep)
        f1, f2, f3, f4 = feats

        # 2. Decoder: do CG-Fuse (training and testing) and WT-Aug(only in training)
        dec_out = self.decoder([f1,f2,f3,f4])  

        # 3. get the final segmentation results
        logits = self.out_conv(dec_out)
        # upsample logits to input resolution
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits



if __name__ == '__main__':
    import os

    root_path = "/home/gxu/proj1/lesionSeg"
    # load dinov3 model
    # set model path
    REPO_DIR = os.path.join(root_path, "dino_seg")
    MODEL_TYPE = ["large", ] # "small", "base", "large"
    DECODER_TYPE = ["cross_guide_wt_unet", ] 
    NUM = 0
    model_weight_path = os.path.join(root_path, "dino_seg/checkpoint/dinov3_convnext_"+MODEL_TYPE[NUM]+"_pretrain_lvd.pth")
    # DINOv3 ConvNeXt models pretrained on web images
    dinov3_convnext = torch.hub.load(REPO_DIR, 'dinov3_convnext_'+MODEL_TYPE[NUM], source='local', weights=model_weight_path)
    # print(dinov3_convnext)


    # load pretrained convnext_tiny from your repo (head=Identity)
    encoder = dinov3_convnext
    encoder.head = nn.Identity()   # remove classifier head
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 768

    for dec in DECODER_TYPE:
        print("Testing decoder:", dec)
        # build segmentation model
        model = DINO_AugSeg(encoder, num_classes=2, model_type=MODEL_TYPE[NUM], decoder_type=dec)   # 1 = binary mask
        # print(model)
        model = model.to(device)
        # model.eval()
        # test forward
        batch_img = torch.randn((1, 3, img_size, img_size))
        batch_img = batch_img.to(device)
        with torch.no_grad():
            pred_mask = model(batch_img)   # [1, 1, 768, 768]
        print(pred_mask.shape)
