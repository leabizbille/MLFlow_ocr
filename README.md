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

projet_ocr/
│
├── api_ocr.py
├── requirements.txt
├── prometheus.yml
└── docker-compose.yml


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


Ton API → expose /metrics
       ↑
   Prometheus → stocke l’historique des métriques
       ↑
   Grafana → lit les données de Prometheus et les affiche joliment

