import mlflow
from Fonctions_OCR import ComparerOriginal_GT, ComparerOriginal_GT_Normaliser
from FonctionsMetrics import compute_metrics  

def evaluer_ocr(img_path: str, ground_truth_path: str, normaliser: bool = True):
    """
    1. Applique OCR avec ComparerOriginal_GT ou ComparerOriginal_GT_Normaliser
    2. Calcule les métriques OCR
    3. Log dans MLflow
    """
    # --- OCR ---
    if normaliser:
        result = ComparerOriginal_GT_Normaliser(img_path, ground_truth_path)
    else:
        result = ComparerOriginal_GT(img_path, ground_truth_path)

    reference = result['reference']       # Texte attendu
    predicted = result['ground_truth']    # Texte OCR

    # --- Début suivi MLflow ---
    mlflow.start_run()
    mlflow.log_param("ocr_correction", False)
    mlflow.log_param("normalisation", normaliser)
    mlflow.log_param("Dataset", img_path.split("/")[-1])  # <-- ajoute le nom du fichier

    # --- Calcul métriques ---
    metrics_before = compute_metrics(reference, predicted)

    # Log longueurs
    mlflow.log_metric("len_predicted", len(predicted))

    # Log métriques
    for k, v in metrics_before.items():
        mlflow.log_metric(f"{k}", v)

    # --- Fin MLflow ---
    mlflow.end_run()

    return {
        "reference": reference,
        "ocr_original": predicted,
        "metrics": metrics_before
    }

# Exemple d'appel
if __name__ == "__main__":
    resultat = evaluer_ocr(
        "Berville_L_CV_IA-avril.jpg",
        "Berville_L_CV_IA-avril.txt",
        normaliser=True
    )
    print(resultat)
