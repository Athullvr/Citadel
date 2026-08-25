# Build from the REPO ROOT, not backend/, since the model + inference code
# live in ../data_collection relative to this file:
#   docker build -f backend/Dockerfile -t agent-cost-predictor-api .
#
# Only copies the inference path (features.py, predict.py, tools.py,
# calibration_data/) -- NOT agent_runner.py/tasks.py/run_collection.py/
# train_model.py (training-only, need the anthropic SDK + API key + full
# runs.jsonl dataset, none of which belong in a production image) and NOT
# the local .venv.

FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py ./main.py

RUN mkdir -p ./data_collection/data ./data_collection/calibration_data
COPY data_collection/features.py ./data_collection/features.py
COPY data_collection/predict.py ./data_collection/predict.py
COPY data_collection/tools.py ./data_collection/tools.py
COPY data_collection/model.joblib ./data_collection/model.joblib
COPY data_collection/calibration_data ./data_collection/calibration_data
COPY data_collection/data/validation_examples.json ./data_collection/data/validation_examples.json

ENV DATA_COLLECTION_DIR=/app/data_collection
ENV PORT=8000
ENV WORKERS=2
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
