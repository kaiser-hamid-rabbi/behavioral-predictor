"""Tiny Transformer with engineered features for behavioral prediction."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x


# Number of hand-engineered numeric features concatenated to pooled output
NUM_ENGINEERED_FEATURES = 6


class BehavioralPredictor(nn.Module):
    """
    Compact sequence model for behavioral prediction.
    
    Input features (per timestep):
      event_name, device_os, channel, category, hour_of_day, traffic_source
    
    Engineered features (per window, concatenated to pooled output):
      purchase_ratio, add_to_cart_ratio, scroll_ratio,
      unique_event_types, avg_time_delta, session_count_norm
    
    Produces 8 concurrent predictions via shared transformer + enriched linear heads.
    """
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        vocab_offsets: dict[str, int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # 6 categorical embeddings per timestep
        self.emb_event    = nn.Embedding(vocab_sizes.get("event_name", 10), d_model, padding_idx=0)
        self.emb_device   = nn.Embedding(vocab_sizes.get("device_os", 10), d_model, padding_idx=0)
        self.emb_channel  = nn.Embedding(vocab_sizes.get("channel", 10), d_model, padding_idx=0)
        self.emb_category = nn.Embedding(vocab_sizes.get("category", 15), d_model, padding_idx=0)
        self.emb_hour     = nn.Embedding(25, d_model, padding_idx=0)  # 0=pad, 1-24=hours
        self.emb_traffic  = nn.Embedding(vocab_sizes.get("traffic_source", 8), d_model, padding_idx=0)

        # Gated fusion: learns optimal combination of 6 embeddings
        self.feature_gate = nn.Linear(d_model * 6, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.pool_norm = nn.LayerNorm(d_model)
        
        # Fuse pooled transformer output with engineered numeric features
        head_input_dim = d_model + NUM_ENGINEERED_FEATURES
        self.fusion_layer = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Prediction Heads
        # Binary heads: simple linear (purchase, churn are well-separated)
        self.head_purchase      = nn.Linear(d_model, 1)
        self.head_churn         = nn.Linear(d_model, 1)
        # Multi-class heads: 2-layer MLP for harder tasks
        self.head_next_event    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, vocab_sizes.get("event_name", 10))
        )
        self.head_channel       = nn.Linear(d_model, vocab_sizes.get("channel", 10))
        # Regression heads
        self.head_engagement    = nn.Linear(d_model, 1)
        self.head_inactivity    = nn.Linear(d_model, 1)
        # Action & period: 2-layer MLP
        self.head_action        = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 20)
        )
        self.head_active_period = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 4)
        )

    def forward(
        self,
        event_ids: torch.Tensor,
        device_ids: torch.Tensor,
        channel_ids: torch.Tensor,
        category_ids: torch.Tensor,
        hour_ids: torch.Tensor,
        traffic_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Categorical inputs: all [batch_size, seq_len]
        numeric_features: [batch_size, NUM_ENGINEERED_FEATURES]
        padding_mask: [batch_size, seq_len]
        Returns: Tuple of 8 prediction tensors
        """
        # Embed 6 categorical features
        e_event   = self.emb_event(event_ids)
        e_device  = self.emb_device(device_ids)
        e_channel = self.emb_channel(channel_ids)
        e_cat     = self.emb_category(category_ids)
        e_hour    = self.emb_hour(hour_ids)
        e_traffic = self.emb_traffic(traffic_ids)
        
        # Gated fusion
        concat = torch.cat([e_event, e_device, e_channel, e_cat, e_hour, e_traffic], dim=-1)
        emb = self.feature_gate(concat)
        emb = emb * math.sqrt(self.d_model)
        
        # Positional encoding
        emb = emb.transpose(0, 1)
        emb = self.pos_encoder(emb)
        emb = emb.transpose(0, 1)
        emb = self.input_norm(emb)
        emb = self.input_dropout(emb)
        
        # Transformer encoding
        encoded = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        
        # Dual pooling: mean + last token
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            mean_pool = (encoded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            mean_pool = encoded.mean(dim=1)
        
        last_token = encoded[:, -1, :]
        pooled = self.pool_norm(mean_pool + last_token)
        
        # Concatenate engineered features + fuse
        enriched = torch.cat([pooled, numeric_features], dim=-1)
        fused = self.fusion_layer(enriched)
             
        # Prediction heads
        return (
            self.head_purchase(fused),
            self.head_churn(fused),
            self.head_next_event(fused),
            self.head_channel(fused),
            self.head_engagement(fused),
            self.head_inactivity(fused),
            self.head_action(fused),
            self.head_active_period(fused),
        )
