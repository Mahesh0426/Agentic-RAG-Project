# Setup & Installation

uv pip install -r requirements.txt --python .venv/bin/python

# Activate virtual environment

source .venv/bin/activate

# Ingest data

python3 -m app.ingestion.processor DATA/noisy_data true

# Run the API server

uvicorn app.main:app --reload --port 8000

# (Alternative without activating venv):

# .venv/bin/uvicorn app.main:app --reload --port 8000

# 3. Start the evaluation UI

streamlit run evals/app.py

.venv/bin/python3 -m app.services.health.connection_checker
