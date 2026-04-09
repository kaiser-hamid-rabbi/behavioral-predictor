# System Architecture

This document describes the architecture designed to scale from the provided dataset subset to the full production dataset (50M+ events, 100K+ users) without code changes.

---

## Multi-Tier Architecture

```
                                        BROWSER (WASM)
                                    ┌─────────────────────┐
                                    │  ONNX Runtime Web   │
                                    │  Sliding Window     │
                                    │  < 10ms inference   │
                                    │  0.85 MB model      │
                                    └────────┬────────────┘
                                             │ events
                                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         SPEED LAYER (Online)                        │
│                                                                     │
│  FastAPI ──► Kafka/Redpanda ──► Consumer ──► PostgreSQL             │
│     │                                            │                  │
│     └──► Redis (Online Feature Store) ◄──────────┘                  │
│              sub-ms feature retrieval                                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        BATCH LAYER (Offline)                        │
│                                                                     │
│  PySpark ETL ──► Parquet ──► Feature Aggregation ──► PostgreSQL     │
│                                                          │          │
│  Celery Worker ──► Load Features ──► Train Model ──► .onnx          │
│                                                                     │
│  RL Feedback ──► Compare predictions vs actuals ──► Online SGD      │
└──────────────────────────────────────────────────────────────────────┘
```

This follows the **Lambda Architecture** pattern. The Speed Layer handles real-time event ingestion and fast feature lookups. The Batch Layer handles heavy computation: distributed ETL, model training, and periodic retraining.

---

## Data Pipeline Scaling

### Tier 1: Single-Machine (Pandas)

For the provided subset or datasets under ~5M events:

```bash
python scripts/train_model.py --sample-users 500
```

The training script reads parquet files directly with Pandas, samples N users, and builds sliding-window sequences in memory. This is the development and evaluation path.

### Tier 2: Distributed (PySpark)

For the full production dataset (50M+ events):

```bash
python scripts/spark_etl_pipeline.py
```

The PySpark pipeline:
- Reads parquet partitions in parallel across Spark executors
- Never loads the full dataset into a single process
- Computes per-user aggregate features (event counts, session metrics, temporal distributions) via distributed `groupBy` and `agg`
- Writes compressed results to PostgreSQL or intermediate parquet
- Runs on local mode (single machine) or cluster mode (YARN, Kubernetes, EMR)

### Tier 3: Streaming (Kafka/Redpanda)

For real-time event ingestion:
- FastAPI pushes incoming events directly to Kafka topics
- This completely bypasses synchronous database inserts
- Kafka acts as a durable shock absorber during traffic spikes
- A background consumer writes events to PostgreSQL in batches
- Zero events are dropped regardless of ingestion rate

---

## Training Pipeline

### Architecture

```
Raw Parquet
    │
    ▼
┌──────────────────┐
│ Load + Sample     │  Pandas/PySpark reads parquet, optionally samples N users
│ (50M rows → N)   │
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Encode            │  Map categoricals to integer IDs via vocabulary
│ event_name → int  │  (scroll=1, add_to_cart=2, purchase=6, etc.)
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Sliding Windows   │  Per-user chronological sort, slide window of 20 events
│ (seq_len=20)      │  Target = the event immediately after the window
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Derive Labels     │  8 targets from actual data:
│                   │  - purchase: 1.0 if next event is "purchase"
│                   │  - churn: 1.0 if gap > 7 days
│                   │  - engagement: ratio of non-scroll in window
│                   │  - next_event, channel, period: from actual next event
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Train (PyTorch)   │  Multi-task loss with:
│                   │  - Focal Loss for imbalanced binary (purchase, churn)
│                   │  - Class-weighted CrossEntropy for multi-class
│                   │  - Task-loss weighting (purchase=3x, churn=2.5x)
│                   │  - Cosine Annealing LR + AdamW + gradient clipping
│                   │  - 85/15 train/val split + early stopping
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Stable Merge      │  Secondary ONNX save reloads and merges external .data 
│ (No .data file)  │  weights into a single self-contained binary.
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Export ONNX       │  opset 15, dynamic batch+seq shapes, 8-input signature
│ (0.85 MB)         │  Compatible with onnxruntime-web (WASM)
└──────────────────┘
```

### Why the Training Scales

1. **User-level parallelism**: Sliding window construction is per-user. Each user's sequence is independent with no cross-user dependencies. This is embarrassingly parallel.
2. **Sampling control**: `--sample-users N` lets you train on any fraction of the data. For development, use 100 users (10s). For production, use all users.
3. **PyTorch DataLoader**: Batched loading with `shuffle=True` and `drop_last=True` ensures constant memory usage regardless of dataset size.
4. **Distributed training ready**: The `BehavioralDataset` class is compatible with PyTorch `DistributedDataParallel` for multi-GPU training.

---

## Model Architecture

### Multi-Task Tiny Transformer

```
Input: [batch, seq_len=20]
    │
    ├── Embedding(event_name, 64) ─┐
    ├── Embedding(device_os, 64)  ─┤
    ├── Embedding(channel, 64)    ─┼── Gated Linear Fusion (Learnable gate)
    ├── Embedding(category, 64)   ─┤
    ├── Embedding(hour, 64)       ─┤
    └── Embedding(traffic, 64)    ─┘
                │
                ▼
    Positional Encoding (sinusoidal)
                │
                ▼
    TransformerEncoder (2 layers, 4 heads, d_model=64, ff=128)
                │
                ▼
    Dual Pooling (Mean + Last Token)
                │
                ▼
    Concat [Pooled Output, 6 Numeric Window Statistics]
                │
                ▼
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
 Linear(64,1) Linear(64,7) Linear(64,4) ... (8 heads total)
 purchase     next_event   period
```

- **Total parameters**: 115,110
- **Exported ONNX size**: 0.29 MB
- All 8 predictions from a single forward pass

### Compression Strategy

| Step | Technique | Effect |
|---|---|---|
| 1 | Stable Merge Logic | Ensures all weights are embedded in the .onnx binary |
| 2 | Opset 15 Downgrade | Guaranteed compatibility across all modern browsers |
| 3 | ONNX Constant Folding | Optimizes static computation at export time |
| Result | | 0.85 MB model, < 10ms browser inference |

The quantizer auto-detects the CPU architecture:
- ARM (Apple Silicon M1/M2/M3, mobile) → `qnnpack` engine
- x86 (Intel/AMD servers, CI) → `fbgemm` engine

---

## Retraining & Feedback Loop

### Scheduled Retraining (Celery)

Celery beat triggers periodic retraining:
1. The worker loads new events from PostgreSQL
2. Rebuilds sliding-window sequences
3. Fine-tunes the existing model (or trains from scratch)
4. Exports a new ONNX artifact
5. The browser fetches the updated model on next page load

### Reinforcement Learning (Contextual Bandits)

The browser sends its predictions back alongside the event stream. A background process (`app/ml/training/rl_feedback.py`) implements a contextual bandit:

1. **Compare**: predicted action vs actual user action
2. **Reward**: +1.0 for correct prediction, -2.0 for churn miss, +0.5 for positive engagement
3. **Update**: Single SGD step against the projection heads (frozen transformer core)

This allows the model to adapt to behavioral shifts between full retraining cycles.

---

## Inference Architecture

### Server-Side (API)

```
POST /predict { user_id, events[] }
    │
    ▼
Redis Feature Store (sub-ms lookup)
    │
    ▼
ONNX Runtime (CPU) → 8 predictions → JSON response
```

### Client-Side (Browser WASM)

```
User clicks "Add to Cart"
    │
    ▼
JS event buffer (last 20 events, in-memory)
    │
    ▼
onnxruntime-web (WASM) → 8 predictions → Update UI
    │
    No network request. < 10ms. Infinite scale.
```

The browser approach eliminates server-side GPU costs entirely. With 1M concurrent users, each user's own device provides the compute.

---

## Database Design

### PostgreSQL (Offline Store)

Events table with range partitioning:

```sql
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    muid UUID NOT NULL,
    event_name VARCHAR(50),
    event_time TIMESTAMPTZ NOT NULL,
    ...
) PARTITION BY RANGE (event_time);

-- Monthly partitions
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

Benefits:
- Queries on recent data only scan the relevant partition
- Old partitions can be archived to cold storage
- No full-table index rebuild as data grows

### Redis (Online Feature Store)

Pre-computed user features cached with TTL:

```
KEY: user_features:{muid}
VALUE: { event_count, last_event_time, purchase_count, ... }
TTL: 24 hours
```

Retrieval follows a tiered pattern:
1. Check Redis (sub-ms)
2. Fall back to PostgreSQL if cache miss
3. Write-through: update Redis on new feature computation
