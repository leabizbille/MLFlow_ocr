# fichier: api_ocr.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import shutil
from pathlib import Path
import os
from fastapi.responses import JSONResponse
from Fonctions_OCR import ComparerOriginal_GT_Normaliser, recuperer_texte_ocr, ComparerOriginal_GT

# Dossier temporaire pour stocker les fichiers uploadés
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


app = FastAPI(
    title="API OCR avec PaddleOCR",
    description="API pour extraire le texte d'une image (tous formats supportés par Pillow) via PaddleOCR",
    version="1.0"
)

@app.post("/ocr/")
async def ocr_image(file: UploadFile = File(...)):
    """
    Reçoit une image (jpg, png, bmp, tiff, etc.), applique l'OCR et retourne le texte extrait.
    """
    # Vérifier le type MIME
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    # Créer un fichier temporaire avec la bonne extension
    suffix = Path(file.filename).suffix  # récupère l'extension (.jpg, .png, etc.)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Appel à la fonction OCR
    try:
        texte = recuperer_texte_ocr(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur OCR: {e}")

    return {
        "filename": file.filename,
        "texte": texte
    }

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API OCR"}

@app.post("/Normaliser le fichier original et le Ground truth/")
async def Comparaison_ocr_Normalise(image: UploadFile = File(...), ground_truth: UploadFile = File(...)):
    """
    Endpoint OCR : retourne le texte OCR et le texte de référence.
    Accepts any image format (jpg, png, bmp, etc.) and ground truth files (txt, doc, docx).
    """
    # --- 1. Sauvegarde fichiers temporairement ---
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    ground_truth_path = os.path.join(UPLOAD_DIR, ground_truth.filename)
    with open(ground_truth_path, "wb") as f:
        shutil.copyfileobj(ground_truth.file, f)

    # --- 2. Appel fonction OCR ---
    try:
        result = ComparerOriginal_GT_Normaliser(image_path, ground_truth_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        # Optionnel : supprimer les fichiers uploadés
        os.remove(image_path)
        os.remove(ground_truth_path)

    # --- 3. Retour JSON ---
    return result


@app.post("/Fichier original et le Ground truth/")
async def Comparaison_ocr(image: UploadFile = File(...), ground_truth: UploadFile = File(...)):
    """
    Endpoint OCR : retourne le texte OCR et le texte de référence.
    Accepts any image format (jpg, png, bmp, etc.) and ground truth files (txt, doc, docx).
    """
    # --- 1. Sauvegarde fichiers temporairement ---
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    ground_truth_path = os.path.join(UPLOAD_DIR, ground_truth.filename)
    with open(ground_truth_path, "wb") as f:
        shutil.copyfileobj(ground_truth.file, f)

    # --- 2. Appel fonction OCR ---
    try:
        result = ComparerOriginal_GT(image_path, ground_truth_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        # Optionnel : supprimer les fichiers uploadés
        os.remove(image_path)
        os.remove(ground_truth_path)

    # --- 3. Retour JSON ---
    return result

