import os
import cv2
import unicodedata
import re

# Gestion des datasets: 
import numpy as np # obligatoire avec PaddleOCR
import pandas as pd
from FonctionsMetrics import compute_metrics, 
from Fonctions_OCR import ComparerOriginal_GT_Normaliser, recuperer_texte_ocr, ComparerOriginal_GT

# ====== Configuration ======
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

nltk.download('punkt')



def main(img, ground_truth_path, output_excel='ocr_metrics_report.xlsx'):
    result = ComparerOriginal_GT(img, ground_truth_path)
    result_Nor = ComparerOriginal_GT_Normaliser(img, ground_truth_path)

    results = []
        for engine, ocr_text in ocr_texts.items():
            # --- Texte brut ---
            metrics_raw = compute_metrics(ground_truth, ocr_text)
            print(f"{engine} [BRUT] : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")
            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': False,
                **metrics_raw,
                'OCR_text': ocr_text
            })

            # --- Texte normalisé ---
            normalized_gt = normalize_text(ground_truth)
            normalized_pred = normalize_text(ocr_text)
            metrics_norm = compute_metrics(normalized_gt, normalized_pred)
            print(f"{engine} [NORMALISE] : CRR={metrics_norm['CRR']}, CER={metrics_norm['CER']}, F1={metrics_norm['F1-score']}")
            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': True,
                **metrics_norm,
                'OCR_text': normalized_pred
            })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nRésultats exportés vers : {output_excel}")




def main(ground_truth_txt_path, pdf_path, output_excel='ocr_metrics_report.xlsx'):
    ground_truth = load_ground_truth_text(ground_truth_txt_path)
    images = pdf_to_images(pdf_path)

    results = []

    for i, pil_img in enumerate(images):
        image_rgb = np.array(pil_img.convert("RGB"))  # ✅ PIL → RGB → numpy array
        print(f"\n--- Page {i + 1} ---")
        ocr_data = extract_ocr_texts(image_rgb)
        for engine, data in ocr_data.items():
            full_text = data['full_text']
            rec_texts = data['texts']
            rec_scores = data['scores']
            rec_bboxes = data['bboxes']

            metrics_raw = compute_metrics(ground_truth, full_text)

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Version': 'brute',
                **metrics_raw,
                'OCR_text': full_text
            })

            print(f"{engine} (brute) : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

            corrected_text = normalize_text(full_text)
            metrics_corr = compute_metrics(ground_truth, corrected_text)

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Version': 'corrigée',
                **metrics_corr,
                'OCR_text': corrected_text
            })

            print(f"{engine} (corrigée) : CRR={metrics_corr['CRR']}, CER={metrics_corr['CER']}, F1={metrics_corr['F1-score']}")

        # Version brute (texte complet concaténé)
        metrics_raw = compute_metrics(ground_truth, full_text)
        results.append({
            'Page': i + 1,
            'OCR_engine': 'PaddleOCR',
            'Version': 'brute',
            **metrics_raw,
            'OCR_text': full_text
        })
        print(f"PaddleOCR (brute) : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

        # Version corrigée (normalisation simple)
        corrected_text = normalize_text(full_text)
        metrics_corr = compute_metrics(ground_truth, corrected_text)
        results.append({
            'Page': i + 1,
            'OCR_engine': 'PaddleOCR',
            'Version': 'corrigée',
            **metrics_corr,
            'OCR_text': corrected_text
        })
        print(f"PaddleOCR (corrigée) : CRR={metrics_corr['CRR']}, CER={metrics_corr['CER']}, F1={metrics_corr['F1-score']}")

        # Optionnel : Tu peux aussi sauvegarder les segments + scores
        # Exemple pour inspection/debug :
        # for t, s in zip(rec_texts, rec_scores):
        #     print(f"Texte: {t} - Score: {s:.3f}")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nRésultats exportés vers : {output_excel}")

    # Générer les graphiques
    plot_ocr_metrics(df)


if __name__ == "__main__":
    GROUND_TRUTH_TXT = "Berville_L_CV_IA-avril.txt"
    PDF_FILE = "Berville_L_CV_IA-avril.jpg"

    main(GROUND_TRUTH_TXT, PDF_FILE)
