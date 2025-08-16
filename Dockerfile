# ==========================
# Dockerfile pour projet OCR
# ==========================

# 1️⃣ Image de base : Python 3.11 slim pour compatibilité packages récents
FROM python:3.11-slim

# 2️⃣ Variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3️⃣ Mise à jour de pip et installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Mise à jour pip
RUN pip install --upgrade pip

# 5️⃣ Copier les fichiers requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Installer PyTorch et torchaudio (CPU ou GPU selon besoins)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7️⃣ Installer PaddleOCR
RUN pip install paddlepaddle paddleocr

# 8️⃣ Installer python-multipart pour l'API
RUN pip install python-multipart

# 9️⃣ Copier le projet dans le container
WORKDIR /app
COPY . /app

# 10️⃣ Exposer les ports
EXPOSE 5000 8000 6006

# 11️⃣ Commande par défaut (exemple : lancer l’API)
CMD ["python", "API.py"]
