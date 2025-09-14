import subprocess
import os

# -----------------------------
# 1️⃣ Lancer MLflow UI
# -----------------------------
mlruns_path = r"C:\Users\Lau\Documents\Moi\1-Travail (sept 23)\3- IA\1- Formation Greta\3- Projets\16-E5Grafana\mlruns"

subprocess.Popen([
    "mlflow", "ui",
    "--backend-store-uri", f"file:///{mlruns_path}",
    "--port", "5000"
])

# -----------------------------
# 2️⃣ Lancer Prometheus
# -----------------------------
prometheus_dir = r"C:\Users\Lau\Documents\Moi\1-Travail (sept 23)\3- IA\1- Formation Greta\3- Projets\Outils\prometheus-3.5.0.windows-amd64"
prometheus_config = os.path.join(prometheus_dir, "prometheus.yml")

# Change de répertoire et lance Prometheus
subprocess.Popen([
    os.path.join(prometheus_dir, "prometheus.exe"),
    f"--config.file={prometheus_config}"
])

# -----------------------------
# 3️⃣ Lancer Uvicorn pour l'API OCR
# -----------------------------
subprocess.Popen([
    "uvicorn", "api_ocr:app",
    "--reload",
    "--port", "8002"
])

print("✅ Tout est lancé. Accède à l'API ici : http://127.0.0.1:8002/docs")
