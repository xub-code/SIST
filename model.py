import torch
import torch.nn as nn
from model_audio import AudioSubNet
from model_text import TextSubNet

def masked_mean(x, mask, dim=1, keepdim=False):
    """
    计算掩码均值 (Robust Masked Mean)
    """
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    m = mask.unsqueeze(-1).float()
    s = (x * m).sum(dim=dim, keepdim=keepdim)
    c = m.sum(dim=dim, keepdim=keepdim).clamp(min=1.0)
    return torch.nan_to_num(s / c)


class CrossAttnSeqToSeq(nn.Module):
    """
    序列到序列的跨模态注意力模块
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_seq, q_mask, kv_seq, kv_mask):
        key_padding_mask = (~kv_mask.bool()) if kv_mask is not None else None
        attn_output, _ = self.mha(
            query=q_seq, key=kv_seq, value=kv_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        # Residual + Norm
        output = self.norm(q_seq + self.dropout(attn_output))
        # Apply Query Mask
        if q_mask is not None:
            output = output * q_mask.unsqueeze(-1).float()
        return torch.nan_to_num(output)


class JointContextRectification(nn.Module):
    """
    [Core Module] Joint Context Rectification (JCR) - Concat Version

    改进点：
    使用 torch.cat 替代相加，保留 4 路特征的完整信息 (Lossless Arbitration)。
    输入维度变为 4 * dim，能捕捉更细粒度的模态间冲突。
    """

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.dim = dim

        # [关键修改] 输入维度变为 dim * 4 (因为是拼接)
        # 瓶颈层维度保持 dim // reduction，这会迫使模型进行更有效的信息筛选
        self.aggregator = nn.Sequential(
            nn.Linear(dim * 4, dim // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        # 输出维度依然是 4 * dim，用于生成 4 个独立的门控权重
        self.arbitrator = nn.Sequential(
            nn.Linear(dim // reduction, dim * 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, a_self, a_cross, t_self, t_cross):
        joint_context = torch.cat([a_self, a_cross, t_self, t_cross], dim=1)
        z = self.aggregator(joint_context)
        weights = self.arbitrator(z)
        w_a_self, w_a_cross, w_t_self, w_t_cross = torch.split(weights, self.dim, dim=1)
        rectified_audio = (a_self * w_a_self) + (a_cross * w_a_cross)
        rectified_text = (t_self * w_t_self) + (t_cross * w_t_cross)
        return torch.cat([rectified_audio, rectified_text], dim=1)


class MultiModalNet(nn.Module):
    def __init__(self, audio_dim, text_dim, num_classes=3, fusion_type="cross_attention", dropout=0.3, shared_dim=256):
        super().__init__()
        self.fusion_type = fusion_type
        self.dropout = nn.Dropout(dropout)

        self.audio_subnet = AudioSubNet(
            input_dim=audio_dim,
            la_heads=4,
            pool_heads=4,
            dropout=dropout
        )

        self.text_subnet = TextSubNet(
            input_dim=text_dim,
            dropout=dropout,
            asp_hidden=256,
            lighting_attn_low_rank=64
        )
        self.audio_proj_pre = nn.Sequential(
            nn.Linear(audio_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )
        self.text_proj_pre = nn.Sequential(
            nn.Linear(text_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )


        if fusion_type == "concat":
            fusion_dim = shared_dim * 2
        elif fusion_type in ["add", "multiply"]:
            fusion_dim = shared_dim
        elif fusion_type == "gated":
            self.gate = nn.Sequential(nn.Linear(shared_dim * 2, shared_dim * 2), nn.Sigmoid())
            fusion_dim = shared_dim * 2
        elif fusion_type == "cross_attention":
            self.seq2seq = CrossAttnSeqToSeq(dim=shared_dim, num_heads=4, dropout=dropout)
            fusion_dim = shared_dim
        elif fusion_type == "bi_cross_attention":
            self.a_from_t = CrossAttnSeqToSeq(dim=shared_dim, num_heads=4, dropout=dropout)
            self.t_from_a = CrossAttnSeqToSeq(dim=shared_dim, num_heads=4, dropout=dropout)
            fusion_dim = shared_dim * 2

        elif fusion_type == "gated_bi_cross_attention":
            self.a_from_t = CrossAttnSeqToSeq(dim=shared_dim, num_heads=4, dropout=dropout)
            self.t_from_a = CrossAttnSeqToSeq(dim=shared_dim, num_heads=4, dropout=dropout)
            self.jcr_module = JointContextRectification(dim=shared_dim, reduction=8)
            fusion_dim = shared_dim * 2
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            self.dropout,
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def forward(self, audio_x, audio_mask, text_x, text_mask):
        needs_seq = self.fusion_type in ["cross_attention", "bi_cross_attention", "gated_bi_cross_attention"]

        if needs_seq:
            audio_seq, audio_feat = self.audio_subnet(audio_x, mask=audio_mask, return_sequence=True)
            text_seq, text_feat = self.text_subnet(text_x, mask=text_mask, return_sequence=True)
        else:
            audio_feat = self.audio_subnet(audio_x, mask=audio_mask)
            text_feat = self.text_subnet(text_x, mask=text_mask)
            audio_seq, text_seq = None, None

        proj_audio = self.audio_proj_pre(audio_feat)
        proj_text = self.text_proj_pre(text_feat)

        audio_seq_shared = None
        text_seq_shared = None
        if needs_seq:
            audio_seq_shared = self.audio_proj_pre(audio_seq)
            text_seq_shared = self.text_proj_pre(text_seq)

        if self.fusion_type == "concat":
            fused = torch.cat([proj_audio, proj_text], dim=1)

        elif self.fusion_type == "add":
            fused = proj_audio + proj_text

        elif self.fusion_type == "multiply":
            fused = proj_audio * proj_text

        elif self.fusion_type == "gated":
            combined = torch.cat([proj_audio, proj_text], dim=1)
            fused = self.gate(combined) * combined

        elif self.fusion_type == "cross_attention":
            a_ctx = self.seq2seq(audio_seq_shared, audio_mask, text_seq_shared, text_mask)
            fused = masked_mean(a_ctx, audio_mask, dim=1) + proj_audio

        elif self.fusion_type == "bi_cross_attention":
            a_ctx = self.a_from_t(audio_seq_shared, audio_mask, text_seq_shared, text_mask)
            a_vec = masked_mean(a_ctx, audio_mask, dim=1) + proj_audio
            t_ctx = self.t_from_a(text_seq_shared, text_mask, audio_seq_shared, audio_mask)
            t_vec = masked_mean(t_ctx, text_mask, dim=1) + proj_text

            fused = torch.cat([a_vec, t_vec], dim=1)

        elif self.fusion_type == "gated_bi_cross_attention":
            a_ctx_seq = self.a_from_t(audio_seq_shared, audio_mask, text_seq_shared, text_mask)
            t_ctx_seq = self.t_from_a(text_seq_shared, text_mask, audio_seq_shared, audio_mask)
            a_cross = masked_mean(a_ctx_seq, audio_mask, dim=1)
            t_cross = masked_mean(t_ctx_seq, text_mask, dim=1)
            fused = self.jcr_module(
                a_self=proj_audio, a_cross=a_cross,
                t_self=proj_text, t_cross=t_cross
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    print("Testing JCR Model (Clean & Concatenated)...")
    model = MultiModalNet(
        audio_dim=768,
        text_dim=768,
        fusion_type="gated_bi_cross_attention",
        shared_dim=256
    )
    B, L = 2, 10
    ax = torch.randn(B, L, 768)
    tx = torch.randn(B, L, 768)
    mask = torch.ones(B, L).bool()

    logits = model(ax, mask, tx, mask)
    print(f"✅ Logits: {logits.shape}")
