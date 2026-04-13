import torch
import torch.nn as nn
import math

'''
Heirarchical Embeddings
'''


class RotaryEmbedding(nn.Module):
    """
    RoPE - no max length needed. Computes position encoding 
    on the fly from whatever indices you pass in.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, positions):
        """
        Args:
            positions: (seq_len,) - arbitrary position indices
        Returns:
            cos, sin each (seq_len, dim)
        """
        # positions can be anything: 0,1,2,3... or 0,1,2,0,1,2,3... 
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to queries/keys."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class HierarchicalEmbedding(nn.Module):
    """
    Decoder input = [CLS_1, CLS_2, ..., CLS_n, tok_1, tok_2, ..., tok_m]
    
    No max lengths. RoPE handles arbitrary segment counts and token lengths.
    Two separate RoPE instances so segment positions and token positions
    live in independent spaces.
    
    Type embedding (2 learned vectors) tells the model what's a CLS vs a token.
    """
    
    def __init__(self, d_model, vocab_size=32000, rope_base=10000):
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # two separate RoPE: one for segment positions, one for token positions
        self.segment_rope = RotaryEmbedding(d_model, base=rope_base)
        self.token_rope = RotaryEmbedding(d_model, base=rope_base)
        
        # type embedding: 0 = sentence, 1 = token
        self.type_embedding = nn.Embedding(2, d_model)
        
        # project encoder CLS to decoder space if needed
        self.cls_proj = nn.Linear(d_model, d_model)
    
    def forward(self, cls_embeddings, token_ids):
        """
        Args:
            cls_embeddings: (batch, num_segments, d_model) - from encoder, any count
            token_ids: (batch, seq_len) - current segment tokens, any length
        
        Returns:
            decoder_input: (batch, num_segments + seq_len, d_model)
            rope_cos: (num_segments + seq_len, d_model)
            rope_sin: (num_segments + seq_len, d_model)
        """
        batch_size, num_segments, _ = cls_embeddings.shape
        _, seq_len = token_ids.shape
        device = cls_embeddings.device
        
        # === CLS side ===
        cls_vectors = self.cls_proj(cls_embeddings)
        cls_vectors = cls_vectors + self.type_embedding(
            torch.zeros(num_segments, dtype=torch.long, device=device)
        )
        
        # === Token side ===
        tok_vectors = self.token_embedding(token_ids)
        tok_vectors = tok_vectors + self.type_embedding(
            torch.ones(seq_len, dtype=torch.long, device=device)
        )
        
        # === Concatenate ===
        decoder_input = torch.cat([cls_vectors, tok_vectors], dim=1)
        
        # === RoPE positions ===
        # segments get their own position space: 0, 1, 2, ..., n-1
        # tokens get their own position space: 0, 1, 2, ..., m-1
        # computed separately, then concatenated
        seg_positions = torch.arange(num_segments, device=device)
        tok_positions = torch.arange(seq_len, device=device)
        
        seg_cos, seg_sin = self.segment_rope(seg_positions)
        tok_cos, tok_sin = self.token_rope(tok_positions)
        
        rope_cos = torch.cat([seg_cos, tok_cos], dim=0)
        rope_sin = torch.cat([seg_sin, tok_sin], dim=0)
        
        return decoder_input, rope_cos, rope_sin


class HierarchicalDecoderLayer(nn.Module):
    """
    Single decoder layer that uses RoPE in its self-attention
    and respects the hierarchical attention mask.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, rope_cos, rope_sin, mask):
        """
        Args:
            x: (batch, seq_len, d_model)
            rope_cos, rope_sin: (seq_len, d_model)
            mask: (seq_len, seq_len) boolean attention mask
        """
        batch, seq_len, d_model = x.shape
        h = self.norm1(x)
        
        q = self.q_proj(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # apply RoPE to queries and keys
        # expand rope to (1, 1, seq_len, head_dim) for broadcasting
        cos = rope_cos[:, :self.head_dim].unsqueeze(0).unsqueeze(0)
        sin = rope_sin[:, :self.head_dim].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # attention with hierarchical mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.o_proj(out)
        x = x + out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class CausalAttentionMask:
    """
    No fixed sizes. Builds mask from whatever counts you pass in.
    
              CLS1 CLS2 CLS3 | tok1 tok2 tok3
    CLS1  [    1    0    0  |   0    0    0  ]
    CLS2  [    1    1    0  |   0    0    0  ]
    CLS3  [    1    1    1  |   0    0    0  ]
    ------+-----------------+----------------
    tok1  [    1    1    1  |   1    0    0  ]
    tok2  [    1    1    1  |   1    1    0  ]
    tok3  [    1    1    1  |   1    1    1  ]
    """
    
    @staticmethod
    def build(num_segments, num_tokens, device="cpu"):
        total = num_segments + num_tokens
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        
        # CLS-to-CLS: causal
        mask[:num_segments, :num_segments] = torch.tril(
            torch.ones(num_segments, num_segments, dtype=torch.bool, device=device)
        )
        
        # Token-to-CLS: full attention
        mask[num_segments:, :num_segments] = True
        
        # Token-to-Token: causal
        mask[num_segments:, num_segments:] = torch.tril(
            torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=device)
        )
        
        return mask


# === Quick test ===
if __name__ == "__main__":
    d_model = 256
    n_heads = 8
    batch_size = 2
    
    # can be ANY count - no max needed
    num_segments = 47
    seq_len = 23
    
    embed = HierarchicalEmbedding(d_model=d_model)
    layer = HierarchicalDecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_model*4)
    
    cls_embs = torch.randn(batch_size, num_segments, d_model)
    tok_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    decoder_input, rope_cos, rope_sin = embed(cls_embs, tok_ids)
    mask = CausalAttentionMask.build(num_segments, seq_len)
    
    output = layer(decoder_input, rope_cos, rope_sin, mask)
    
    print(f"Segments: {num_segments}, Tokens: {seq_len} (no max limit)")
    print(f"Decoder input: {decoder_input.shape}")
    print(f"Output: {output.shape}")
    
    # works with completely different sizes too
    num_segments_2 = 1000
    seq_len_2 = 5
    
    cls_embs_2 = torch.randn(batch_size, num_segments_2, d_model)
    tok_ids_2 = torch.randint(0, 32000, (batch_size, seq_len_2))
    
    decoder_input_2, rope_cos_2, rope_sin_2 = embed(cls_embs_2, tok_ids_2)
    mask_2 = CausalAttentionMask.build(num_segments_2, seq_len_2)
    
    output_2 = layer(decoder_input_2, rope_cos_2, rope_sin_2, mask_2)
    print(f"\nSegments: {num_segments_2}, Tokens: {seq_len_2} (still no max limit)")
    print(f"Decoder input: {decoder_input_2.shape}")
    print(f"Output: {output_2.shape}")