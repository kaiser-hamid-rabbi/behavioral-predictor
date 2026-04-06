"""Tiny Transformer implementation for sequence classification/regression."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple

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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class BehavioralPredictor(nn.Module):
    """
    Compact sequence model for behavioral prediction.
    Outputs multiple predictions via different heads.
    """
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        vocab_offsets: dict[str, int], # For combining embeddings correctly
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # In a real model, we might use separate embeddings and concat or add.
        # Adding them keeps d_model small (e.g. 64)
        
        # We need sum of all vocab sizes for a single unified embedding layer,
        # or separate embeddings added together. We use separate added embeddings for simplicity.
        self.emb_event = nn.Embedding(vocab_sizes.get("event_name", 100), d_model, padding_idx=0)
        self.emb_device = nn.Embedding(vocab_sizes.get("device_os", 10), d_model, padding_idx=0)
        self.emb_channel = nn.Embedding(vocab_sizes.get("channel", 10), d_model, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Prediction Heads
        # Purchase (Binary)
        self.head_purchase = nn.Linear(d_model, 1)
        # Churn (Binary)
        self.head_churn = nn.Linear(d_model, 1)
        # Next Event (Multiclass)
        self.head_next_event = nn.Linear(d_model, vocab_sizes.get("event_name", 100))
        # Channel (Multiclass)
        self.head_channel = nn.Linear(d_model, vocab_sizes.get("channel", 10))
        # Engagement (Regression)
        self.head_engagement = nn.Linear(d_model, 1)
        # Inactivity risk (Regression)
        self.head_inactivity = nn.Linear(d_model, 1)
        # Recommended Action (Multiclass - assumed 20 actions)
        self.head_action = nn.Linear(d_model, 20)
        # Active Period (Multiclass - 4 periods)
        self.head_active_period = nn.Linear(d_model, 4)

    def forward(
        self,
        event_ids: torch.Tensor,
        device_ids: torch.Tensor,
        channel_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        """
        Inputs: 
            *_ids: [batch_size, seq_len]
            padding_mask: [batch_size, seq_len] with True for paddings (ignoring them)
        Returns: Tuple of all predictions
        """
        # Embeddings + Addition
        emb = self.emb_event(event_ids) + self.emb_device(device_ids) + self.emb_channel(channel_ids)
        emb = emb * math.sqrt(self.d_model)
        
        # Transformer assumes [batch, seq, feature] if batch_first=True
        # Positional encoder expects [seq, batch, feature] by default in pytorch tutorial, 
        # but since we made batch_first=True we need to adjust
        emb = emb.transpose(0, 1) # [seq, batch, feature] for PE compatibility
        emb = self.pos_encoder(emb)
        emb = emb.transpose(0, 1) # back to [batch, seq, feature]
        
        # Encode
        # src_key_padding_mask requires True for padded positions
        encoded = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        
        # Pool to a single sequence vector representation (e.g. mean pooling over non-padded)
        if padding_mask is not None:
             # Mask out padded positions
             mask = (~padding_mask).unsqueeze(-1).float()
             pooled = (encoded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
             pooled = encoded.mean(dim=1)
             
        # Heads
        # Using sigmoid for binary, softmax for multiclass during training loss computation usually relies on raw logits. 
        # We output logits/raw values to pair with BCEWithLogitsLoss / CrossEntropyLoss / MSELoss
        p_purchase = self.head_purchase(pooled)
        p_churn = self.head_churn(pooled)
        p_next_event = self.head_next_event(pooled)
        p_channel = self.head_channel(pooled)
        p_engagement = self.head_engagement(pooled)
        p_inactivity = self.head_inactivity(pooled)
        p_action = self.head_action(pooled)
        p_active_period = self.head_active_period(pooled)
        
        return (
            p_purchase, 
            p_churn, 
            p_next_event, 
            p_channel, 
            p_engagement, 
            p_inactivity, 
            p_action, 
            p_active_period
        )
