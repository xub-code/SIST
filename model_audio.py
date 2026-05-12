# model_audio.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def xavier_linear_(m):
    """Xavier 初始化线性层"""
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LinearAttention(nn.Module):
    """
    线性注意力 (Linear Attention): O(T) 复杂度
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.2, eps=1e-6):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(xavier_linear_)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        # 1. Projections
        Q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        # 2. Kernel Function (ELU + 1)
        Qp = F.elu(Q) + 1.0
        Kp = F.elu(K) + 1.0

        # 3. Apply Mask to K and V
        if mask is not None:
            mask_f = mask.unsqueeze(1).unsqueeze(-1).float()
            Kp = Kp * mask_f
            V = V * mask_f

        # 4. Efficient Attention Calculation
        # O(T) aggregation: K^T * V
        S = torch.matmul(Kp.transpose(-2, -1), V)
        z = Kp.sum(dim=-2)  # Normalization factor

        numer = torch.matmul(Qp, S)
        denom = torch.matmul(Qp, z.unsqueeze(-1)).squeeze(-1) + self.eps

        Y = numer / denom.unsqueeze(-1)
        Y = Y.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        Y = self.out_proj(self.dropout(Y))

        return Y


class MultiHeadQueryPooling(nn.Module):
    """
    标准版 Attention Pooling: 使用可学习的 Query 进行池化
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.3):
        super().__init__()
        # Learnable Query Token: [1, 1, D]
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.apply(xavier_linear_)

    def forward(self, x, mask=None):
        B = x.shape[0]
        # Expand query to batch size: [B, 1, D]
        q = self.query_token.expand(B, -1, -1)
        q = self.norm(q)

        # PyTorch MHA expects key_padding_mask=True for ignored positions
        key_padding_mask = (~mask.bool()) if mask is not None else None

        out, _ = self.mha(query=q, key=x, value=x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class AudioSubNet(nn.Module):
    def __init__(self, input_dim, la_heads=4, pool_heads=4, dropout=0.3):
        super().__init__()
        self.pre_ln = nn.LayerNorm(input_dim)
        self.linear_attn = LinearAttention(input_dim, num_heads=la_heads, dropout=0.1)
        self.pool = MultiHeadQueryPooling(input_dim, num_heads=pool_heads, dropout=dropout)

    def forward(self, x, mask=None, return_sequence=False):
        x = self.pre_ln(x)
        # 提取序列特征
        Y = self.linear_attn(x, mask)
        # 提取全局向量
        feat = self.pool(Y, mask)

        if return_sequence:
            return Y, feat
        return feat


if __name__ == "__main__":
    B, T, D = 8, 128, 768
    x = torch.randn(B, T, D)
    lengths = torch.randint(low=T // 4, high=T, size=(B,))
    mask = torch.arange(T).unsqueeze(0).repeat(B, 1) < lengths.unsqueeze(1)
    mask = mask.to(torch.bool)

    model = AudioSubNet(input_dim=D, dropout=0.3)

    feat = model(x, mask)
    print("✅ [Audio] 纯净池化特征形状:", feat.shape)
    y_seq, feat = model(x, mask, return_sequence=True)
    print("✅ [Audio] 时序特征形状:", y_seq.shape)