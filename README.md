# Behavioral Prediction System

A production-grade, distributed pipeline built to ingest massive behavioral event streams and deploy ultra-compact multi-task Transformers directly into a user's browser for zero-latency, scale-independent real-time inference.

## Concept & Capabilities

Predicting real-time behavioral vectors (churn risk, purchase probability, next anticipated event) conventionally demands heavy server-side GPU operations that paralyze budgets under significant traffic. 

This project solves massive scaling by distributing the computational load **directly into the client's browser** via the ONNX Runtime WebAssembly (WASM) backend, limiting our backend responsibility strictly to asynchronous high-throughput event ingestion and periodic background model retraining.

### Predicted Behaviors
Through a unified sequence-aware attention layer, we output the following concurrently:
- Purchase Probability (Binary)
- Churn Risk (Binary)
- Lookahead Next Event (Categorical)
- Preferred Channel Engagement (Categorical)
- Aggregate Engagement Score (Regression)
- Inactivity Risk (Regression)
- Optimal Recommended Action (Categorical)
- Peak Active User Period (Categorical)

## Repository Structure

```
behavioral-predictor/
├── backend/                  # Python 3.11 FastAPI & PyTorch ML Engine
│   ├── alembic/              # Async Database Migrations
│   ├── app/                  # Core Microservices (Ingestion, RL, Prediction)
│   ├── scripts/              # Bulk Parquet/Spark ETL Loading
│   └── tests/                # Pytest Coverage Suite
├── browser_model/            # Vanilla JS / WebAssembly Edge Client
│   └── js/predictor.js       # Localized Browser Inference Core
├── docs/                     # Architectural Documentation
│   └── architecture.md       # Details on scaling to 50M+ payloads
└── docker-compose.yml        # Unified Multi-container Orchestration
```

## System Architecture

Our solution ensures **horizontal scalability** out of the box through aggressive decoupling, asynchronous message queues, and temporal database partitioning.

> See [docs/architecture.md](docs/architecture.md) for full flowcharts encompassing our continuous data pipelines, Redis feature stores, and reinforcement learning strategies.

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (Optional, if running backend locally without containers)

### 1. Launching the Infrastructure
The fastest way to test ingestion boundaries and the frontend demo is via the managed docker stack.

```bash
docker-compose up -d --build
```

**Services initialized:**
- **API Server**: http://localhost:8000/docs
- **Client Simulator UI**: http://localhost:3000
- **PostgreSQL 16**: (Port 5432)
- **Redis Broker**: (Port 6379)
- **Celery Training Worker**: Background daemon.

### 2. Validating Browser Edge Inference
Navigate to http://localhost:3000. 

You will find a completely isolated Web Client. Trigger actions manually using the UI buttons to populate an event buffer. The ONNX WASM engine intercepts this buffer and calculates all 8 behavior targets instantly (< 10ms) without ever executing a network request to the backend.

### 3. Local Development & ML Pipeline 
If you wish to test the compression techniques or automated retraining loops:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Boot the isolated test suite
pytest tests/ -v

# Run the strict model packager (Verifies sub 1.0 MB limits)
python scripts/train_model.py --check-size
```

## Built With
* **Machine Learning**: PyTorch 2.1, ONNX
* **Browser Runtime**: ONNXRuntime-Web (WASM)
* **Backend Core**: FastAPI, SQLAlchemy 2.0 (Async), Celery
* **Storage Layers**: PostgreSQL (Time Partitioned), Redis
