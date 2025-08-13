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

Ton API → expose /metrics
       ↑
   Prometheus → stocke l’historique des métriques
       ↑
   Grafana → lit les données de Prometheus et les affiche joliment

