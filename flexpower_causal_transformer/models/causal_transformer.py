import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, film_in, d_model):
        super().__init__()
        self.gamma = nn.Linear(film_in, d_model)
        self.beta  = nn.Linear(film_in, d_model)

    def forward(self, x, film_vec):
        # x: [B, L, D], film_vec: [B, film_in]
        gamma = self.gamma(film_vec).unsqueeze(1)  # [B,1,D]
        beta  = self.beta(film_vec).unsqueeze(1)   # [B,1,D]
        return x * (1 + gamma) + beta

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # split heads
        def split(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1,2)  # [B, h, L, d]
        q, k, v = split(q), split(k), split(v)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, L, L]

        # causal mask: allow attending to <= current position
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # broadcastable additive mask

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # [B, h, L, d]
        out = out.transpose(1,2).contiguous().view(B, L, D)
        out = self.out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout1(self.attn(self.ln1(x), attn_mask))
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x

class CausalTransformerFlex(nn.Module):
    def __init__(self, num_cont, d_model=128, n_heads=4, n_layers=3, dropout=0.1,
                 use_film=True, film_dim=64,
                 num_stations=64, num_receivers=64, num_antennas=64, num_constellations=4, num_prns=64):
        super().__init__()
        self.d_model = d_model
        # project continuous channels to d_model
        self.input_proj = nn.Linear(num_cont, d_model)

        # categorical embeddings
        self.emb_station = nn.Embedding(num_stations, film_dim)
        self.emb_receiver = nn.Embedding(num_receivers, film_dim)
        self.emb_antenna = nn.Embedding(num_antennas, film_dim)
        self.emb_constellation = nn.Embedding(num_constellations, film_dim)
        self.emb_prn = nn.Embedding(num_prns, film_dim)
        self.use_film = use_film
        film_in = film_dim * 5
        self.film = FiLM(film_in, d_model) if use_film else None

        # positional encoding (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, d_model))  # max seq len 4096
        nn.init.normal_(self.pos_emb, std=0.02)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # point-level logit

    def forward(self, x, meta):
        # x: [B, L, C]; meta: dict of [B,1] ids
        B, L, C = x.shape
        h = self.input_proj(x)  # [B, L, D]
        h = h + self.pos_emb[:, :L, :]

        if self.use_film:
            station = self.emb_station(meta['station_id'].squeeze(-1))
            receiver = self.emb_receiver(meta['receiver_id'].squeeze(-1))
            antenna = self.emb_antenna(meta['antenna_id'].squeeze(-1))
            constellation = self.emb_constellation(meta['constellation_id'].squeeze(-1))
            prn = self.emb_prn(meta['prn_id'].squeeze(-1))
            film_vec = torch.cat([station, receiver, antenna, constellation, prn], dim=-1)
            h = self.film(h, film_vec)

        # transformer
        for blk in self.blocks:
            h = blk(h)

        h = self.ln_final(h)
        # readout at last time step (current epoch t)
        ht = h[:, -1, :]  # [B, D]
        logit = self.head(ht)  # [B, 1]
        return logit.squeeze(-1)
