# metrics_server.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request
from starlette.responses import Response
import time

# --- Définition des métriques ---
REQUEST_COUNT = Counter(
    "ocr_api_requests_total", "Nombre total de requêtes reçues", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "ocr_api_request_duration_seconds", "Durée des requêtes OCR en secondes", ["endpoint"]
)
TEXT_LENGTH = Gauge(
    "ocr_text_length", "Longueur du texte OCR renvoyé", ["endpoint"]
)

# --- Middleware pour mesurer ---
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Enregistre la durée
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)

    # Enregistre le nombre de requêtes
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()

    return response

# --- Endpoint /metrics ---
async def metrics_endpoint(request: Request):
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
