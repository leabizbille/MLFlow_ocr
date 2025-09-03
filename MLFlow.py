import mlflow
import optuna
from Fonctions_OCR import ComparerOriginal_GT, ComparerOriginal_GT_Normaliser
from FonctionsMetrics import compute_metrics
from pathlib import Path

# --- Configuration MLflow ---
MLFLOW_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("OCR_Evaluation")

# --- Fonction d'évaluation simplifiée ---
def evaluer_ocr_modele(img_path, gt_path, normaliser=True):
    # OCR + comparaison
    if normaliser:
        result = ComparerOriginal_GT_Normaliser(img_path, gt_path)
    else:
        result = ComparerOriginal_GT(img_path, gt_path)

    predicted = result['reference']
    reference = result['ground_truth']

    # Logging MLflow
    with mlflow.start_run():
        mlflow.log_param("normalisation", normaliser)
        mlflow.log_param("Dataset", Path(img_path).name)

        metrics = compute_metrics(reference, predicted)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_artifact(img_path)
        mlflow.log_artifact(gt_path)

    return metrics

# --- Optuna Objective simplifiée ---
def objective(trial):
    normaliser = trial.suggest_categorical("normaliser", [True, False])
    metrics = evaluer_ocr_modele(
        "Berville_L_CV_IA-avril.jpg",
        "Berville_L_CV_IA-avril.txt",
        normaliser=normaliser
    )
    return metrics.get("f1", 0)

# --- Lancer Optuna ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best params:", study.best_params)
    print("Best value (F1):", study.best_value)
