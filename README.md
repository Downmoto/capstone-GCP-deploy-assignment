# heart failure prediction model deployment

this repository contains an end-to-end machine learning deployment project using the kaggle heart failure prediction dataset.

## project structure

- `training/train.py`: trains and evaluates two models, then saves artifacts
- `src/main.py`: fastapi app with `/health` and `/predict`
- `src/security.py`: api-key check using `x-api-key`
- `artifacts/`: saved preprocessor and best model
- `Dockerfile`: container image definition
- `scripts/deploy_cloud_run.sh`: gcp cloud run deployment helper
- `scripts/sample_request.json`: sample payload for testing

## dataset setup

1. run dataset download helper (uses kagglehub):

```bash
source .venv/bin/activate
python scripts/download_dataset.py
```

2. this creates `data/heart.csv` automatically.

## local environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## train and save artifacts

```bash
python training/train.py
```

expected output files:

- `artifacts/model.joblib`
- `artifacts/preprocessor.joblib`
- `artifacts/metrics.json`

## run api locally

```bash
export API_KEY="dev-secret-key"
uvicorn src.main:app --reload --port 8080
```

health check:

```bash
curl http://127.0.0.1:8080/health
```

prediction request:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-secret-key" \
  -d @scripts/sample_request_01.json
```

## run in docker

```bash
docker build -t heart-failure-api:latest .
docker run --rm -p 8080:8080 -e API_KEY="dev-secret-key" heart-failure-api:latest
```

## api contract

request body:

```json
{
  "data": [
    {
      "age": 40,
      "sex": "M",
      "chest pain type": "ATA",
      "resting bp s": 140,
      "cholesterol": 289,
      "fasting bs": 0,
      "resting ecg": "Normal",
      "max hr": 172,
      "exercise angina": "N",
      "oldpeak": 0.0,
      "st slope": "Up"
    }
  ]
}
```

response body:

```json
{
  "predictions": [0],
  "probabilities": [0.13]
}
```
