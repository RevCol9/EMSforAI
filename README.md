# EMSforAI

A minimal FastAPI backend scaffold for an Equipment Management System (EMS) demo with AI-friendly hooks. The current setup provides:

- Conda environment specification (`environment.yml`).
- FastAPI app with health and placeholder AI recommendation endpoints.
- Basic SQLAlchemy models and CRUD routes for device models and devices (SQLite by default).

## Getting started

### 1) Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate emsforai
```

### 2) Configure environment variables
Copy the example file and adjust if needed (e.g., switch to PostgreSQL):
```bash
cp .env.example .env
```

### 3) Initialize and run the API
```bash
# Run with reload for local development
fastapi dev backend/main.py
# or
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will auto-create the SQLite database file (`emsforai.db`) on first run.

### API quick check
- `GET /health` – service heartbeat.
- `GET /recommendations` – sample AI suggestion placeholder.
- `POST /devices/models` – create a device model.
- `GET /devices/models` – list device models.
- `POST /devices` – create a device linked to a model.
- `GET /devices` – list devices.

### Next steps
- Extend schemas/models for telemetry, maintenance tasks, and lifecycle events.
- Add migrations via Alembic.
- Integrate real AI/ML inference in `backend/ai_service.py`.
