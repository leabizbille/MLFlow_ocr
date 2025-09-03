import pytest
from fastapi.testclient import TestClient
import os
import sys
from PIL import Image
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from API import app
from Fonctions_OCR import ComparerOriginal_GT, ComparerOriginal_GT_Normaliser
from Fonctions_clean import normalize_text

client = TestClient(app)


# -----------------------
# Tests FastAPI Endpoints
# -----------------------

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API OCR"}

def test_ocr_endpoint_with_image():
    test_image = "tests/Berville_L_CV_IA-avril.jpg"  # mettre une petite image de test
    with open(test_image, "rb") as f:
        response = client.post("/ocr/", files={"file": ("Berville_L_CV_IA-avril.jpg", f, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "texte" in data
    assert data["filename"] == "Berville_L_CV_IA-avril.jpg"

def test_ocr_endpoint_with_invalid_file():
    test_file = "tests/Berville_L_CV_IA-avril.txt"
    with open(test_file, "rb") as f:
        response = client.post("/ocr/", files={"file": ("Berville_L_CV_IA-avril.txt", f, "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Le fichier doit être une image."

def test_normaliser_endpoint(tmp_path):
    # Créer une vraie petite image
    test_image = tmp_path / "Berville_L_CV_IA-avril.png"
    img = Image.new('RGB', (1, 1), color='white')
    img.save(test_image)

    # Fichier ground truth
    test_gt = tmp_path / "sample_gt.txt"
    test_gt.write_text("Ceci est un texte de test pour le Ground Truth.")

    with open(test_image, "rb") as img_f, open(test_gt, "rb") as gt_f:
        response = client.post(
            "/Normaliser le fichier original et le Ground truth/",
            files={
                "image": (test_image.name, img_f, "image/png"),
                "ground_truth": (test_gt.name, gt_f, "text/plain")
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert "reference" in data
    assert "ground_truth" in data

# -----------------------
# Tests fonctions OCR
# -----------------------
def test_normalize_text_basic():
    text = "Élève très motivé!!!"
    normalized = normalize_text(text)
    assert normalized == "eleve tres motive"

def test_comparer_original_gt_returns_dict(tmp_path):
    # Créer une image 1x1 blanche
    img_file = tmp_path / "fake_img.png"
    img = Image.new('RGB', (1, 1), color='white')
    img.save(img_file)

    # Fichier ground truth
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("Texte attendu")

    result = ComparerOriginal_GT(str(img_file), str(gt_file))
    assert isinstance(result, dict)
    assert "reference" in result
    assert "ground_truth" in result

@patch("Fonctions_OCR.PaddleOCR")
def test_comparer_original_gt_returns_dict(mock_ocr, tmp_path):
    # Mock OCR.predict pour retourner un texte fictif
    mock_ocr.return_value.predict.return_value = [{"rec_texts": ["texte OCR simulé"]}]

    # Créer un fichier image factice (le contenu n'a pas d'importance)
    img_file = tmp_path / "fake_img.png"
    img_file.write_bytes(b"fake image content")

    # Fichier ground truth
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text("Texte attendu")

    result = ComparerOriginal_GT(str(img_file), str(gt_file))
    assert isinstance(result, dict)
    assert result["reference"] == "texte OCR simulé"
    assert result["ground_truth"] == "Texte attendu"
