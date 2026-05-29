# Enterprise AI Microservice Architecture (FastAPI & Scikit-Learn)

A production-grade Python AI microservice designed with a .NET-inspired clean architecture layout. This project decouples the deterministic web serving framework from the stochastic machine learning asset lifecycle.

## Architectural Highlights
* **App Factory Pattern**: Decoupled environment routing configuration using `pydantic-settings`.
* **Lifespan Initialization Hook**: Synchronous ML binary deserialized into shared state memory precisely *once* at system startup to prevent memory leaks.
* **Vectorized Data Transformation**: Real-time feature engineering normalization optimized via C-speed SIMD operations inside NumPy arrays.
* **Thread-Safe Concurrent Execution**: Heavy synchronous mathematical calculations offloaded to worker threads via native FastAPI thread pool configuration.

## Setup & Execution

### 1. Provision Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Compile/Train the AI Binary

```bash
python -m src.pipelines.train
```

### 3. Run the Web Service Host

```bash
python -m uvicorn src.app.main:app --reload
```

### 4. Run the Automation Test Suite

``` bash
PYTHONPATH=. python -m pytest tests/
```


