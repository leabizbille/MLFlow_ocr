

import subprocess

# Lance MLflow UI avec les bons arguments
subprocess.Popen([
    "mlflow", "ui",
    "--backend-store-uri", "file:///C:/Users/Lau/Documents/Moi/1-Travail (sept 23)/3- IA/1- Formation Greta/3- Projets/16-E5Grafana/mlruns",
    "--port", "5000"
])


cd "C:\Users\Lau\Documents\Moi\1-Travail (sept 23)\3- IA\1- Formation Greta\3- Projets\Outils\prometheus-3.5.0.windows-amd64"
.\prometheus.exe --config.file=prometheus.yml
uvicorn api_ocr:app --reload --port 8000

http://127.0.0.1:8002/docs
