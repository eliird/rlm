import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BERTEmbedding(nn.Module):
    """
    BERT Embedding: token + positional + segment embeddings.
    """

    def __init__(self, vocab_size, embed_size, max_len=512, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float) * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.pe[:, :sequence.size(1)] + self.segment(segment_label)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using F.scaled_dot_product_attention
    (auto-dispatches to Flash Attention / memory-efficient kernels).
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Fused QKV projection
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # F.scaled_dot_product_attention handles masking, scaling, softmax, dropout
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual"""

    def __init__(self, hidden, attn_heads, ff_hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = MultiHeadAttention(hidden, attn_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = FeedForward(hidden, ff_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class BERT(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers.
    Returns hidden states for all positions; use [:, 0] for CLS embedding.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.embedding = BERTEmbedding(vocab_size, hidden, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, segment_info):
        # Padding mask: [batch, 1, 1, seq_len] for broadcasting over heads and query positions
        mask = (x > 0).unsqueeze(1).unsqueeze(2)

        x = self.embedding(x, segment_info)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)