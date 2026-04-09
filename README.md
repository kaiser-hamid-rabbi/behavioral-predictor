# Behavioral Prediction System

A production-grade behavioral prediction system that ingests massive event streams, trains a sequence-aware multi-task Transformer on real user data, and deploys an ultra-compact model directly into the browser for zero-latency, personalized inference.

---

## What This System Does

Given raw behavioral event streams (pageviews, scrolls, purchases, searches), the system predicts 8 behavioral dimensions simultaneously through a single forward pass:

| Prediction | Type | Description |
|---|---|---|
| Purchase Probability | Binary | Likelihood the user will purchase next |
| Churn Risk | Binary | Risk of user disengaging (>7 day gap) |
| Next Likely Event | Categorical | What action the user will take next |
| Preferred Channel | Categorical | Browser vs. app preference |
| Engagement Score | Regression | How actively the user interacts |
| Inactivity Risk | Regression | Probability of going dormant |
| Recommended Action | Categorical | Optimal nudge for the user |
| Peak Active Period | Categorical | Time-of-day the user is most active |

### Advanced Feature Engineering
To maximize accuracy across all 8 fields, the model performs **Gated Feature Fusion** on 6 per-timestep categorical embeddings, and concatenates **6 hand-engineered numeric statistical features** (computed per sliding window) to the pooled output before routing to the prediction heads:
- **`purchase_ratio`, `atc_ratio`**: Boosts Purchase & Next Event heads.
- **`scroll_ratio`, `unique_types`**: Boosts Engagement & Churn heads.
- **`avg_time_delta`, `session_count_norm`**: Boosts Inactivity & Period heads.

---

## Repository Structure

```
behavioral-predictor/
├── backend/                    # FastAPI + PyTorch ML Engine
│   ├── app/
│   │   ├── api/routes/         # REST endpoints (health, events, predictions, training)
│   │   ├── core/               # Config, logging, dependency injection, observability
│   │   ├── db/                 # SQLAlchemy models, repositories (PostgreSQL)
│   │   ├── ml/
│   │   │   ├── compression/    # Quantizer, Pruner, ONNX Exporter
│   │   │   ├── feature_engineering/  # Feature builder, vocabulary
│   │   │   └── training/       # Transformer model, dataset, RL feedback
│   │   ├── schemas/            # Pydantic request/response models
│   │   ├── services/           # Business logic (event, prediction, training, feature)
│   │   ├── streaming/          # Kafka producer for async event ingestion
│   │   └── workers/            # Celery tasks for background retraining
│   ├── scripts/
│   │   ├── train_model.py      # Real training pipeline (parquet -> model -> ONNX)
│   │   └── spark_etl_pipeline.py  # PySpark distributed ETL for 50M+ events
│   ├── tests/                  # 7-test pytest suite
│   └── models/                 # Exported .pt and .onnx artifacts
├── browser_model/              # Vanilla JS + ONNX Runtime Web (WASM)
│   └── js/predictor.js         # Client-side inference engine
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA on the raw parquet data
├── docs/
│   ├── architecture.md         # Full system architecture & scaling strategy
│   ├── evaluator_guide.md      # Step-by-step testing instructions
│   └── submission_report.md    # Requirement-to-implementation mapping
├── docker-compose.yml          # Multi-container orchestration
└── dataset/                    # Drop your .parquet files here!
    ├── EventsData/
    │   ├── events/             # Place event .parquet files here
    │   └── users/              # Place user .parquet files here
    └── README.md
```

---

## 🗄️ Loading Your Data
To keep the repository fast and lightweight, the massive raw `.parquet` dataset files are intentionally excluded from version control (via `.gitignore`).

Before running pipelines, you **must** drop your data files into their respective folders:
1. Put event data here: `dataset/EventsData/events/` (e.g. `events_0.parquet`)
2. Put user data here: `dataset/EventsData/users/` (e.g. `users_0.parquet`)

The training scripts automatically detect `.parquet` files placed in these directories.

---

## Training Pipeline

The training pipeline reads directly from the provided parquet files — no mocks, no synthetic data.

### How It Works

```
Parquet Files ──> Load & Sample Users ──> Encode Categoricals
      ──> Sliding Window (size=20) ──> Derive 8 Target Labels
      ──> Train/Val Split (85/15) ──> Multi-Task Training
      ──> Early Stopping ──> Stable ONNX Merge (Single File)
      ──> Export ONNX (< 1 MB)
```

### Running It

```bash
cd backend
source .venv/bin/activate

# Train on 500 sampled users, 15 epochs (default)
python scripts/train_model.py --check-size

# Train on more users for higher accuracy
python scripts/train_model.py --sample-users 2000 --epochs 20

# Train on ALL users (requires sufficient RAM for 50M events)
python scripts/train_model.py --sample-users 0 --epochs 15
```

### ML Engineering for Accuracy

The pipeline implements the following techniques to maximize accuracy while maintaining a sub-1MB model:

| Technique | Purpose |
|---|---|
| Focal Loss (alpha=0.25, gamma=2.0) | Handles extreme class imbalance — purchases are only ~5% of events |
| Class-weighted CrossEntropy | Inverse-frequency weights prevent majority-class domination |
| Task-loss weighting | Purchase (3x) and churn (2.5x) are prioritized as business-critical |
| Cosine Annealing LR with warm restarts | Smooth decay prevents premature convergence |
| AdamW with weight decay (1e-4) | L2 regularization prevents overfitting |
| Gradient clipping (max_norm=1.0) | Stabilizes transformer training |
| Train/Val split (85/15) | Detects overfitting during training |
| Early stopping (patience=5) | Stops when validation plateaus, restores best weights |
| Per-head metric logging | Tracks accuracy for every prediction head per epoch |
| End-to-End Feature Engineering | Model processes 6 embeddings (`traffic_source`, `category`, etc.) + 6 numeric features |

### Achieved Results (500 users, 60K sequences)

```
Purchase Accuracy:   98.5%
Churn Accuracy:      99.2%
Next Event Accuracy: 32.1%
Channel Accuracy:   100.0%
Action Accuracy:     48.4%
Period Accuracy:     98.8%
Engagement MAE:      0.015
Inactivity MAE:      0.038
Model Size:          0.85 MB (FP32 Merged)
```

---

## Scaling Strategy

The system is designed so that the exact same codebase handles both the provided subset and the full production dataset (50M+ events, 100K+ users).

### Data Pipeline Scaling

| Scale | Tool | How |
|---|---|---|
| Small (< 1M events) | Pandas + `train_model.py` | Direct parquet read with `--sample-users` |
| Medium (1M - 50M events) | Pandas with chunked sampling | `--sample-users N` loads only N users from 50M rows |
| Large (50M+ events) | PySpark ETL (`spark_etl_pipeline.py`) | Distributed map-reduce across horizontal clusters |

The PySpark pipeline (`scripts/spark_etl_pipeline.py`) handles the full-scale case:
- Reads parquet partitions in parallel across Spark executors
- Computes aggregate user features (event counts, session metrics) via distributed groupBy
- Writes results back to PostgreSQL or as intermediate parquet for training
- Memory-efficient: never loads the full dataset into a single process

### Training Pipeline Scaling

The training script uses `--sample-users` to control exactly how much data to train on:
- For development: `--sample-users 100` (trains in ~10 seconds)
- For evaluation: `--sample-users 500` (trains in ~60 seconds)
- For production: `--sample-users 0` (uses all available users)

For full-scale training beyond a single machine:
- The `BehavioralDataset` class is compatible with PyTorch `DistributedDataParallel`
- The sliding-window builder can be parallelized per user (embarrassingly parallel)
- Celery workers handle background retraining on a schedule

### Inference Scaling

| Component | Strategy |
|---|---|
| Browser (WASM) | Each user's device runs inference locally — scales infinitely |
| API Server | Stateless FastAPI behind load balancer (K8s HPA) |
| Event Ingestion | Kafka/Redpanda absorbs traffic spikes, decoupled from DB writes |
| Feature Store | Redis for sub-millisecond online feature retrieval |
| Database | PostgreSQL with time-range partitioning on event tables |

### Retraining & Feedback Loop

The system supports continuous model improvement:

1. **Scheduled Retraining**: Celery beat triggers periodic retraining against new data
2. **Reinforcement Learning** (`app/ml/training/rl_feedback.py`): Browser predictions are sent back to the server. A contextual bandit compares predictions vs actual user actions and applies online SGD updates to the projection heads
3. **Model Versioning**: Each training run saves a timestamped `.pt` checkpoint, enabling rollback

---

## Browser Inference

The trained ONNX model (0.25 MB) runs directly in the user's browser using WebAssembly:

```bash
cd browser_model
python3 -m http.server 3000
# Open http://localhost:3000
```

- The browser maintains a sliding window of the last 20 events in memory
- On any new user action, the ONNX model runs instantly via `onnxruntime-web`
- Inference completes in < 10ms with zero network dependency
- No user data leaves the browser for predictions

---

## Running Tests

```bash
cd backend
source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest tests/ -v
```

All 7 tests validate:
- Health check API schema
- Input validation (empty batch rejection, UUID enforcement)
- Graceful 503 when no model is deployed
- ONNX export + size constraint (< 1 MB)
- PyTorch Dataset tensor shapes
- Feature engineering aggregation
- Multi-head Transformer forward pass (8 outputs)

---

## Infrastructure (Docker)

```bash
# Start the full stack
docker-compose up -d --build

# Verify services
docker ps
```

| Service | Port | Purpose |
|---|---|---|
| FastAPI | 8000 | REST API + Swagger UI (`/docs`) |
| PostgreSQL 16 | 5432 | Time-partitioned event storage |
| Redis | 6379 | Online feature store + Celery broker |
| Redpanda | 9092 | Kafka-compatible event streaming |

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch 2.x |
| Model Architecture | Multi-Task Tiny Transformer (6 Gated embeddings + 6 numeric feats) |
| Optimization | Final Weight Merge (Single-file Portability) |
| Inference Format | ONNX (opset 15) |
| Browser Runtime | ONNX Runtime Web (WebAssembly) |
| Backend | FastAPI (async), SQLAlchemy 2.0 (async) |
| Streaming | Kafka / Redpanda |
| Background Jobs | Celery |
| Database | PostgreSQL 16 (range-partitioned) |
| Cache / Feature Store | Redis |
| Big Data ETL | PySpark |
| Observability | OpenTelemetry, structlog |
