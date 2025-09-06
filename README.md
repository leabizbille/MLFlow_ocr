# MLFlow_ocr

Téléchargement de Pytorch pour les modeles :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams
Sous windows : pip install paddlepaddle

Pour envoyer des fichier dans l api : pip install python-multipart

# ML FLOW
mlflow ui
http://127.0.0.1:5000/#/experiments

API docs
http://127.0.0.1:8000/docs#/

Prometheus :
http://localhost:8000/metrics

16-E5Grafana/
│
├── API.py
├── requirements.txt
├── prometheus.yml
|── docker-compose.yml
├── Dockerfile
├── Fonctions_clean.py
├── Fonctions_OCR.py
├── FonctionsMetrics.py
├── metrics_server.py
├── MLFlow.py
├── prometheus.yml
├── README.md
└── requirements.txt


docker-compose up --build

docker version
 Cloud integration: v1.0.35+desktop.11
 Version:           25.0.3
 API version:       1.44
 Go version:        go1.21.6
 Git commit:        4debf41
 Built:             Tue Feb  6 21:13:02 2024
 OS/Arch:           windows/amd64
 Context:           default


 docker build -t api_ocr:v1 .


Docker
Ton API → expose /metrics
       ↑
   Prometheus → stocke l’historique des métriques
       ↑
   Grafana → lit les données de Prometheus et les affiche joliment


# schéma conceptuel du flux

[Client / User] 
      |
      v
[FastAPI - API OCR]  ------------------------+
      |                                     |
      | Upload image / Ground Truth         |
      v                                     |
[Tmp Storage / Uploads]                     |
      |                                     |
      +--> [Fonctions_OCR.py]               |
              - ComparerOriginal_GT         |
              - ComparerOriginal_GT_Normaliser
              - recuperer_texte_ocr        |
      |                                     |
      v                                     |
[Texte OCR]                                 |
      |                                     |
      +--> [Metrics Server]                 |
              - compute_metrics            |
              - plot_ocr_metrics           |
      |                                     |
      v                                     |
[Prometheus] <-- Middleware FastAPI ------> [Grafana]
      |
      v
(Métriques collectées: REQUEST_COUNT, REQUEST_LATENCY, TEXT_LENGTH)

FastAPI : endpoints /ocr/, /Normaliser le fichier original et le Ground truth/, /Fichier original et le Ground truth/

PaddleOCR : extraction de texte OCR

Fonctions_OCR.py : gestion OCR et normalisation

metrics_server.py : calcul et monitoring des métriques OCR

MLflow + Transformers : correction et optimisation du texte OCR

Prometheus : collecte des métriques temps réel

Grafana : visualisation des métriques


Explications des parties clés

CI : build-test

Vérifie le code à chaque push/PR.

Installe Python + dépendances.

Lint (flake8) + tests (pytest) + couverture.

CD : docker-build-deploy

Dépend de build-test (ne s’exécute que si CI réussit).

Build l’image Docker avec le SHA du commit.

Push sur Docker Hub (ou tout registry).

Déploie sur ton serveur via SSH et docker-compose up -d.

Secrets GitHub à créer :

DOCKER_USERNAME et DOCKER_PASSWORD → pour Docker Hub.

SERVER_SSH → user@host avec clé SSH pour déploiement.

Placement du fichier :

Tout fichier .yml de workflow doit être dans .github/workflows/ci-cd.yml.

GitHub Actions détecte automatiquement tous les fichiers dans ce dossier.


Pytest :
pytest tests/test_api.py --maxfail=1 --disable-warnings -q


docker build -t ocr-api:latest .
docker compose up -d
docker compose up -d mlflow

docker ps
docker logs -f mlflow

docker build -t ocr-optuna .




docker run --rm --network 16-e5grafana_default -v "C:/Users/Lau/Documents/Moi/1-Travail (sept 23)/3-IA/1-Formation Greta/3-Projets/16-E5Grafana/mlruns:/app/mlruns" -p 5000:5000 ocr-optuna
