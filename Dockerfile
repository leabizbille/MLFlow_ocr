# ==========================
# Dockerfile pour projet OCR
# ==========================

FROM python:3.11-slim

# Installer les dépendances système nécessaires pour OpenCV et PaddleOCR
# Installer les dépendances système nécessaires pour OpenCV et PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Variables d'environnement Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Mise à jour pip
RUN pip install --upgrade pip

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installer PyTorch (CPU) et PaddleOCR
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install paddlepaddle paddleocr
RUN pip install python-multipart

# Copier le projet
WORKDIR /app
COPY . /app

# Exposer les ports
EXPOSE 5000 8000 6006

# Commande par défaut
CMD ["python", "API.py"]
