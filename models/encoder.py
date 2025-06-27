# csi_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# (MLP, QKV, MHA, Encoder, CrossEncoder etc... unchanged)

# --- Paste your full AutoEncoder class here ---
# Simplified usage: We’ll only use the encoder portion of it

class MLP(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4*dim)
        self.fc2 = nn.Linear(4*dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class QKV(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.heads = heads

    def forward(self, x):
        B, N, C = x.shape
        q = self.w_q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = self.w_k(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = self.w_v(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        return q, k, v

class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(q.shape[0], -1, self.heads * (q.shape[-1]))
        return self.proj(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = QKV(dim, heads)
        self.mha = MHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        q, k, v = self.qkv(self.norm1(x))
        x = x + self.mha(q, k, v)
        x = x + self.mlp(self.norm2(x))
        return x

class CrossEncoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv1 = QKV(dim, heads)
        self.mha1 = MHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = MLP(dim)

        self.norm3 = nn.LayerNorm(dim)
        self.qkv2 = QKV(dim, heads)
        self.mha2 = MHA(dim, heads)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = MLP(dim)

    def forward(self, x, y):
        qx, kx, vx = self.qkv1(self.norm1(x))
        qy, ky, vy = self.qkv2(self.norm3(y))
        x = x + self.mha1(qx, ky, vy)
        y = y + self.mha2(qy, kx, vx)
        x = x + self.mlp1(self.norm2(x))
        y = y + self.mlp2(self.norm4(y))
        return x, y


class CSIEncoder(nn.Module):
    def __init__(self, seq=64, dim=32, heads=4, codeword=512, depth=4):
        super().__init__()
        self.seq = seq
        self.dim = dim
        self.codeword = codeword

        self.input_proj = nn.Linear(2, dim)  # Assuming input shape is (B, 2, 32, 32)

        self.transformer = nn.Sequential(*[
            EncoderBlock(dim=dim, heads=heads) for _ in range(depth)
        ])

        self.output_proj = nn.Linear(dim, codeword )  # So total output = (B, seq, codeword//seq) ⇒ flattened to (B, codeword)

    def forward(self, x):
        B, C, H, W = x.shape  # x: (B, 2, 32, 32)
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, 2) → (B, seq=1024, 2)
        x = self.input_proj(x)  # (B, 1024, dim)
        x = self.transformer(x)  # (B, 1024, dim)
        x = self.output_proj(x)  # (B, 1024, codeword // seq)
        x = x.mean(dim=1)  # Flatten to (B, codeword)
        return x
