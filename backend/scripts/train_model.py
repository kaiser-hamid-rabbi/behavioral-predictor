"""
Production Training Pipeline with Full Feature Engineering.

Engineered features per window:
  1. purchase_ratio    - fraction of window events that are "purchase" (purchase head)
  2. add_to_cart_ratio - fraction that are "add_to_cart" (purchase intent signal)
  3. scroll_ratio      - fraction that are "scroll" (low engagement signal)
  4. unique_event_types - normalized count of distinct event types (diversity)
  5. avg_time_delta    - average seconds between consecutive events (activity pace)
  6. session_count_norm - number of distinct sessions normalized (session depth)

Categorical features per timestep:
  event_name, device_os, channel, category, hour_of_day, traffic_source

Usage:
    python scripts/train_model.py --check-size
    python scripts/train_model.py --epochs 20 --sample-users 1000
"""

import argparse
import glob
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from app.ml.training.model import BehavioralPredictor
from app.ml.training.dataset import BehavioralDataset
from app.ml.compression.quantizer import Quantizer
from app.ml.compression.pruner import Pruner
from app.ml.compression.exporter import OnnxExporter


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Vocabulary mappings
# ---------------------------------------------------------------------------
EVENT_NAMES = ["<PAD>", "scroll", "add_to_cart", "viewcontent", "pageview", "search", "purchase"]
DEVICE_OS   = ["<PAD>", "android", "ios", "desktop"]
CHANNELS    = ["<PAD>", "browser", "app"]
CATEGORIES  = [
    "<PAD>", "books", "beauty", "electronics", "fashion", "sports",
    "home", "toys", "grocery", "auto", "health", "music", "games", "other"
]
TRAFFIC_SRC = ["<PAD>", "direct", "organic", "paid", "referral", "social"]

EVENT_TO_ID    = {v: i for i, v in enumerate(EVENT_NAMES)}
DEVICE_TO_ID   = {v: i for i, v in enumerate(DEVICE_OS)}
CHANNEL_TO_ID  = {v: i for i, v in enumerate(CHANNELS)}
CATEGORY_TO_ID = {v: i for i, v in enumerate(CATEGORIES)}
TRAFFIC_TO_ID  = {v: i for i, v in enumerate(TRAFFIC_SRC)}

VOCAB_SIZES = {
    "event_name":      len(EVENT_NAMES),
    "device_os":       len(DEVICE_OS),
    "channel":         len(CHANNELS),
    "category":        len(CATEGORIES),
    "traffic_source":  len(TRAFFIC_SRC),
}

SEQ_LENGTH = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_events(data_dir: str, sample_users: int | None = None) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"  Found {len(files)} parquet partition(s)")
    cols = ["muid", "event_name", "event_time", "device_os", "channel",
            "category", "traffic_source", "session_id"]
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total raw events loaded: {len(df):,}")

    if sample_users:
        unique_users = df["muid"].unique()
        chosen = np.random.choice(unique_users, size=min(sample_users, len(unique_users)), replace=False)
        df = df[df["muid"].isin(chosen)]
        print(f"  Sampled to {len(chosen)} users ({len(df):,} events)")

    return df


def hour_to_period(hour: int) -> int:
    if hour < 6:    return 0
    elif hour < 12: return 1
    elif hour < 18: return 2
    else:           return 3


# ---------------------------------------------------------------------------
# Feature engineering: per-window numeric features
# ---------------------------------------------------------------------------
def compute_window_features(window: list[dict], seq_length: int) -> list[float]:
    """
    Compute 6 hand-engineered numeric features from a sliding window.
    These capture aggregate behavioral patterns that help every prediction head.
    """
    event_names = [e["event_name"] for e in window]
    
    # 1. Purchase ratio: fraction of events that are purchases
    #    → Helps purchase head (strong purchase intent if user recently purchased)
    purchase_ratio = sum(1 for e in event_names if e == "purchase") / seq_length
    
    # 2. Add-to-cart ratio: fraction of add_to_cart events
    #    → Strong purchase intent signal; add_to_cart often precedes purchase
    atc_ratio = sum(1 for e in event_names if e == "add_to_cart") / seq_length
    
    # 3. Scroll ratio: fraction of scroll events
    #    → Low engagement signal (scrolling without interacting)
    #    → Helps engagement and churn heads
    scroll_ratio = sum(1 for e in event_names if e == "scroll") / seq_length
    
    # 4. Unique event types: how diverse are the user's recent actions
    #    → Higher diversity → more engaged user, less likely to churn
    unique_types = len(set(event_names)) / len(EVENT_NAMES)  # normalized 0-1
    
    # 5. Average time delta: mean seconds between consecutive events
    #    → Fast actions = high engagement; slow = potential churn
    times = [e["event_time"] for e in window]
    deltas = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
    avg_delta = np.mean(deltas) if deltas else 0.0
    # Normalize: log-scale to handle wide range (seconds to days)
    avg_delta_norm = min(np.log1p(avg_delta) / 15.0, 1.0)  # cap at 1.0
    
    # 6. Session count: how many distinct sessions in the window
    #    → Multiple sessions = returning user, less churn risk
    sessions = set(e.get("session_id", "") for e in window)
    session_count_norm = min(len(sessions) / 5.0, 1.0)  # normalize, cap at 1.0
    
    return [
        purchase_ratio,
        atc_ratio,
        scroll_ratio,
        unique_types,
        avg_delta_norm,
        session_count_norm,
    ]


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame) -> list[dict]:
    """Build sliding-window training records with full feature engineering."""
    records = []

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.sort_values(["muid", "event_time"])

    df["event_id_enc"]    = [EVENT_TO_ID.get(v, 0) for v in df["event_name"]]
    df["device_id_enc"]   = [DEVICE_TO_ID.get(v, 0) for v in df["device_os"]]
    df["channel_id_enc"]  = [CHANNEL_TO_ID.get(v, 0) for v in df["channel"]]
    df["category"]        = df["category"].fillna("")
    df["category_id_enc"] = [CATEGORY_TO_ID.get(str(v).lower().strip(), 0) for v in df["category"]]
    df["traffic_source"]  = df["traffic_source"].fillna("")
    df["traffic_id_enc"]  = [TRAFFIC_TO_ID.get(str(v).lower().strip(), 0) for v in df["traffic_source"]]
    df["hour"]            = df["event_time"].dt.hour
    df["hour_enc"]        = df["hour"] + 1  # 0 = padding

    user_count = 0
    for _, group in df.groupby("muid"):
        events = group.to_dict("records")
        if len(events) < SEQ_LENGTH + 1:
            continue

        for i in range(len(events) - SEQ_LENGTH):
            window = events[i : i + SEQ_LENGTH]
            target_event = events[i + SEQ_LENGTH]

            # Categorical sequences
            event_ids    = [e["event_id_enc"] for e in window]
            device_ids   = [e["device_id_enc"] for e in window]
            channel_ids  = [e["channel_id_enc"] for e in window]
            category_ids = [e["category_id_enc"] for e in window]
            hour_ids     = [e["hour_enc"] for e in window]
            traffic_ids  = [e["traffic_id_enc"] for e in window]
            padding_mask = [False] * SEQ_LENGTH

            # Engineered numeric features
            numeric_features = compute_window_features(window, SEQ_LENGTH)

            # --- Targets ---
            target_next_event = target_event["event_id_enc"]
            target_channel    = target_event["channel_id_enc"]
            target_purchase   = 1.0 if target_event["event_name"] == "purchase" else 0.0

            last_time = window[-1]["event_time"]
            next_time = target_event["event_time"]
            gap_hours = (next_time - last_time).total_seconds() / 3600
            target_churn = 1.0 if gap_hours > 168 else 0.0

            non_scroll = sum(1 for e in window if e["event_name"] != "scroll")
            target_engagement = non_scroll / SEQ_LENGTH

            window_span = max((window[-1]["event_time"] - window[0]["event_time"]).total_seconds() / 3600, 0.01)
            target_inactivity = min(window_span / SEQ_LENGTH, 1.0)

            target_action = min(target_next_event, 19)
            target_active_period = hour_to_period(target_event["hour"])

            records.append({
                "event_name":      event_ids,
                "device_os":       device_ids,
                "channel":         channel_ids,
                "category":        category_ids,
                "hour":            hour_ids,
                "traffic_source":  traffic_ids,
                "padding_mask":    padding_mask,
                "numeric_features": numeric_features,
                "target_purchase": target_purchase,
                "target_churn": target_churn,
                "target_next_event": target_next_event,
                "target_channel": target_channel,
                "target_engagement": target_engagement,
                "target_inactivity": target_inactivity,
                "target_recommended_action": target_action,
                "target_active_period": target_active_period,
            })
        user_count += 1

    print(f"  Built {len(records):,} training sequences from {user_count:,} users")
    return records


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(records: list[dict], key: str, num_classes: int) -> torch.Tensor:
    counts = Counter(r[key] for r in records)
    total = sum(counts.values())
    weights = torch.ones(num_classes)
    for cls_id, count in counts.items():
        if cls_id < num_classes:
            weights[cls_id] = total / (num_classes * max(count, 1))
    return weights.clamp(min=0.1, max=10.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    total = 0
    correct = {k: 0 for k in ["purchase", "churn", "next_event", "channel", "action", "period"]}
    total_eng_err = 0.0
    total_inact_err = 0.0

    for batch in dataloader:
        preds = model(
            batch["event_name"], batch["device_os"], batch["channel"],
            batch["category"], batch["hour"], batch["traffic_source"],
            batch["numeric_features"], batch["padding_mask"],
        )
        p_purchase, p_churn, p_next_event, p_channel, \
            p_engagement, p_inactivity, p_action, p_active_period = preds

        bs = batch["event_name"].size(0)
        total += bs

        correct["purchase"]   += ((torch.sigmoid(p_purchase) > 0.5).float() == batch["target_purchase"]).sum().item()
        correct["churn"]      += ((torch.sigmoid(p_churn) > 0.5).float() == batch["target_churn"]).sum().item()
        correct["next_event"] += (p_next_event.argmax(dim=1) == batch["target_next_event"]).sum().item()
        correct["channel"]    += (p_channel.argmax(dim=1) == batch["target_channel"]).sum().item()
        correct["action"]     += (p_action.argmax(dim=1) == batch["target_recommended_action"]).sum().item()
        correct["period"]     += (p_active_period.argmax(dim=1) == batch["target_active_period"]).sum().item()
        total_eng_err         += (p_engagement - batch["target_engagement"]).abs().sum().item()
        total_inact_err       += (p_inactivity - batch["target_inactivity"]).abs().sum().item()

    model.train()
    return {
        "purchase_acc":    correct["purchase"] / max(total, 1) * 100,
        "churn_acc":       correct["churn"] / max(total, 1) * 100,
        "next_event_acc":  correct["next_event"] / max(total, 1) * 100,
        "channel_acc":     correct["channel"] / max(total, 1) * 100,
        "action_acc":      correct["action"] / max(total, 1) * 100,
        "period_acc":      correct["period"] / max(total, 1) * 100,
        "engagement_mae":  total_eng_err / max(total, 1),
        "inactivity_mae":  total_inact_err / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 5e-4,
    grad_clip: float = 1.0,
    patience: int = 7,
    class_weights: dict | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 3, 2), T_mult=2
    )

    focal = FocalLoss(alpha=0.25, gamma=2.0)
    mse = nn.MSELoss()

    ce_event   = nn.CrossEntropyLoss(weight=class_weights.get("next_event"), label_smoothing=0.1)
    ce_channel = nn.CrossEntropyLoss(weight=class_weights.get("channel"),    label_smoothing=0.05)
    ce_action  = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_period  = nn.CrossEntropyLoss(weight=class_weights.get("period"),     label_smoothing=0.1)

    tw = {
        "purchase": 3.0, "churn": 2.5,
        "next_event": 3.0, "channel": 1.0,
        "engagement": 1.5, "inactivity": 1.0,
        "action": 2.0, "period": 2.0,
    }

    best_val_score = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            preds = model(
                batch["event_name"], batch["device_os"], batch["channel"],
                batch["category"], batch["hour"], batch["traffic_source"],
                batch["numeric_features"], batch["padding_mask"],
            )
            p_purchase, p_churn, p_next_event, p_channel, \
                p_engagement, p_inactivity, p_action, p_active_period = preds

            loss = (
                tw["purchase"]   * focal(p_purchase, batch["target_purchase"])
                + tw["churn"]    * focal(p_churn, batch["target_churn"])
                + tw["next_event"] * ce_event(p_next_event, batch["target_next_event"])
                + tw["channel"]  * ce_channel(p_channel, batch["target_channel"])
                + tw["engagement"] * mse(p_engagement, batch["target_engagement"])
                + tw["inactivity"] * mse(p_inactivity, batch["target_inactivity"])
                + tw["action"]   * ce_action(p_action, batch["target_recommended_action"])
                + tw["period"]   * ce_period(p_active_period, batch["target_active_period"])
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        val = evaluate(model, val_loader)

        # Composite score: weighted average of error rates
        val_score = (
            0.3 * (100 - val["next_event_acc"]) / 100
            + 0.2 * (100 - val["purchase_acc"]) / 100
            + 0.15 * (100 - val["period_acc"]) / 100
            + 0.15 * (100 - val["action_acc"]) / 100
            + 0.1 * val["engagement_mae"]
            + 0.1 * val["inactivity_mae"]
        )

        print(
            f"  Epoch {epoch+1:02d}/{epochs}  "
            f"Loss:{avg_loss:.3f}  LR:{current_lr:.5f}  │  "
            f"Purch:{val['purchase_acc']:.1f}%  "
            f"Churn:{val['churn_acc']:.1f}%  "
            f"NxtEvt:{val['next_event_acc']:.1f}%  "
            f"Chan:{val['channel_acc']:.1f}%  "
            f"Act:{val['action_acc']:.1f}%  "
            f"Period:{val['period_acc']:.1f}%  "
            f"EngMAE:{val['engagement_mae']:.3f}  "
            f"InactMAE:{val['inactivity_mae']:.3f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        print("  Restored best model weights.")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train behavioral prediction model")
    parser.add_argument("--check-size", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sample-users", type=int, default=500)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        for c in ["../dataset/EventsData/events/", "../../dataset/EventsData/events/"]:
            if os.path.isdir(c) and glob.glob(os.path.join(c, "*.parquet")):
                data_dir = c
                break
    if not data_dir:
        print("ERROR: Could not find events parquet directory.")
        return

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1
    print("\n[1/6] Loading event data from parquet files...")
    sample = args.sample_users if args.sample_users > 0 else None
    df = load_events(data_dir, sample_users=sample)

    # Step 2
    print("\n[2/6] Building sequences with feature engineering...")
    t0 = time.time()
    records = build_sequences(df)
    print(f"  Took {time.time() - t0:.1f}s")
    if not records:
        print("ERROR: No sequences built.")
        return

    # Step 3
    print("\n[3/6] Computing class weights...")
    class_weights = {
        "next_event": compute_class_weights(records, "target_next_event", VOCAB_SIZES["event_name"]),
        "channel":    compute_class_weights(records, "target_channel", VOCAB_SIZES["channel"]),
        "period":     compute_class_weights(records, "target_active_period", 4),
    }
    purchase_pos = sum(1 for r in records if r["target_purchase"] > 0.5)
    churn_pos = sum(1 for r in records if r["target_churn"] > 0.5)
    print(f"  Purchase: {purchase_pos:,} positive ({purchase_pos/len(records)*100:.1f}%)")
    print(f"  Churn:    {churn_pos:,} positive ({churn_pos/len(records)*100:.1f}%)")
    event_dist = Counter(r["target_next_event"] for r in records)
    print(f"  Event distribution: { {EVENT_NAMES[k]: v for k, v in sorted(event_dist.items()) if k < len(EVENT_NAMES)} }")

    # Show sample engineered features
    sample_feats = records[0]["numeric_features"]
    feat_names = ["purchase_ratio", "atc_ratio", "scroll_ratio", "unique_types", "avg_time_delta", "session_count"]
    print(f"  Sample engineered features: { {n: round(v, 3) for n, v in zip(feat_names, sample_feats)} }")

    # Step 4
    print("\n[4/6] Training Multi-Task Transformer (6 embeddings + 6 numeric features)...")
    dataset = BehavioralDataset(records)
    val_size = max(int(len(dataset) * 0.15), 1)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = BehavioralPredictor(
        vocab_sizes=VOCAB_SIZES, vocab_offsets={},
        d_model=64, nhead=4, num_layers=2, dim_ff=128,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Train: {train_size:,}  |  Val: {val_size:,}\n")

    t0 = time.time()
    model = train(model, train_loader, val_loader,
                  epochs=args.epochs, lr=args.lr,
                  patience=args.patience, class_weights=class_weights)
    print(f"  Training took {time.time() - t0:.1f}s")

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║       FINAL VALIDATION METRICS           ║")
    print("  ╠══════════════════════════════════════════╣")
    final = evaluate(model, val_loader)
    for k, v in final.items():
        unit = "%" if "acc" in k else ""
        print(f"  ║  {k:>20s}: {v:>7.2f}{unit:1s}      ║")
    print("  ╚══════════════════════════════════════════╝")

    pt_path = str(out_dir / "behavioral_predictor.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"\n  Saved checkpoint: {pt_path}")

    # Step 5 & 6: Unified Export
    # We export the evaluated model directly. It is already ~300KB (FP32), 
    # well under the 1MB target, ensuring maximum stability in ORT Web.
    onnx_path = str(out_dir / "behavioral_predictor.onnx")
    size_bytes = OnnxExporter.export_model(model, onnx_path, seq_length=SEQ_LENGTH)
    mb = size_bytes / (1024 * 1024)
    print(f"\n[5/6] Exporting ONNX to {onnx_path}...")
    print(f"  Final ONNX Model Size: {mb:.2f} MB")

    if args.check_size:
        assert mb < 1.0, f"Model size {mb:.2f}MB exceeds target"
        print("  Size check PASSED.")

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
