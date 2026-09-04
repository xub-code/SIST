# model_text.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def xavier_linear_(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def masked_mean(x, mask):
    if mask is None:
        return x.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2, low_rank_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.q_low = nn.Linear(self.head_dim, low_rank_dim, bias=False)
        self.k_low = nn.Linear(self.head_dim, low_rank_dim, bias=False)

        # Dynamic Weighting Gate
        self.alpha_proj = nn.Linear(embed_dim, num_heads)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.zeros_(self.alpha_proj.bias)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.apply(xavier_linear_)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        Q_norm = F.normalize(self.q_low(Q), dim=-1)
        K_norm = F.normalize(self.k_low(K), dim=-1)
        score_lr = torch.matmul(Q_norm, K_norm.transpose(-2, -1))

        score_orig = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)

        x_mean = masked_mean(x, mask)
        alpha = torch.sigmoid(self.alpha_proj(x_mean)).view(B, H, 1, 1)

        scores = alpha * score_lr + (1 - alpha) * score_orig

        if mask is not None:
            scores = scores.masked_fill(~mask.view(B, 1, 1, T), float('-inf'))

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        Y = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, D)
        Y = self.out_proj(self.dropout(Y))
        return Y


class AttentiveStatsPool(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.2, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.proj = nn.Linear(2 * d_model, d_model)
        self.apply(xavier_linear_)

    def forward(self, x, mask=None, eps: float = 1e-6):
        logits = self.scorer(x).squeeze(-1) / self.temperature
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(logits, dim=-1).unsqueeze(1)  # [B, 1, T]
        mu = torch.bmm(attn, x).squeeze(1)
        ex2 = torch.bmm(attn, x * x).squeeze(1)
        sd = torch.sqrt((ex2 - mu ** 2).clamp_min(eps))

        return self.proj(torch.cat([mu, sd], dim=-1))


class TextSubNet(nn.Module):
    def __init__(self, input_dim, dropout=0.3, asp_hidden=256, lighting_attn_low_rank=64):
        super().__init__()
        self.pre_ln = nn.LayerNorm(input_dim)
        self.lighting_attn_v2 = Attention(
            embed_dim=input_dim, num_heads=4, dropout=dropout,
            low_rank_dim=lighting_attn_low_rank
        )
        self.pool = AttentiveStatsPool(d_model=input_dim, hidden=asp_hidden, dropout=dropout)

    def forward(self, x, mask=None, return_sequence=False):
        x = self.pre_ln(x)
        Y = self.lighting_attn_v2(x, mask)
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
    model = TextSubNet(input_dim=D, dropout=0.3)

    feat = model(x, mask)
    print("✅ [Text] 纯净池化特征形状:", feat.shape)
    y_seq, feat = model(x, mask, return_sequence=True)
    print("✅ [Text] 时序特征形状:", y_seq.shape)