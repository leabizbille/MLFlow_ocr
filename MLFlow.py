import mlflow
import optuna
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from Fonctions_OCR import ComparerOriginal_GT, ComparerOriginal_GT_Normaliser
from FonctionsMetrics import compute_metrics

# --- Initialisation des modèles ---
models_info = {
    "t5-small": {
        "tokenizer": AutoTokenizer.from_pretrained("t5-small"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    },
    "sshleifer/distilbart-cnn-12-6": {
        "tokenizer": AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    }
}

# --- Fonction d'évaluation ---
def evaluer_ocr_modele(img_path, gt_path, model_name, normaliser=True, num_beams=4, max_new_tokens=256):
    tokenizer = models_info[model_name]["tokenizer"]
    model = models_info[model_name]["model"]
    
    corr_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU uniquement
    )

    # --- OCR ---
    if normaliser:
        result = ComparerOriginal_GT_Normaliser(img_path, gt_path)
    else:
        result = ComparerOriginal_GT(img_path, gt_path)

    predicted = result['reference']
    reference = result['ground_truth']

    # Tronquer si trop long
    max_len = tokenizer.model_max_length
    if len(predicted) > max_len:
        predicted = predicted[:max_len]

    # --- MLflow ---
    mlflow.start_run()
    mlflow.log_param("model", model_name)
    mlflow.log_param("normalisation", normaliser)
    mlflow.log_param("num_beams", num_beams)
    mlflow.log_param("max_new_tokens", max_new_tokens)
    mlflow.log_param("Dataset", img_path.split("/")[-1])

    # --- Correction ---
    predicted_corrige = corr_pipeline(
        f"corrige: {predicted}",
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        early_stopping=True
    )[0]['generated_text']

    # --- Calcul des metrics ---
    metrics_before = compute_metrics(reference, predicted)
    metrics_after = compute_metrics(reference, predicted_corrige)

    # Logging métriques
    for k, v in metrics_before.items():
        mlflow.log_metric(f"before_{k}", v)
    for k, v in metrics_after.items():
        mlflow.log_metric(f"after_{k}", v)

    mlflow.end_run()

    return metrics_after

# --- Optuna Objective ---
def objective(trial):
    model_name = trial.suggest_categorical("model_name", list(models_info.keys()))
    normaliser = trial.suggest_categorical("normaliser", [True, False])
    num_beams = trial.suggest_int("num_beams", 2, 6)
    max_new_tokens = trial.suggest_int("max_new_tokens", 64, 512, step=64)

    # Retourner une métrique à maximiser/minimiser (ex: F1)
    metrics = evaluer_ocr_modele(
        "Berville_L_CV_IA-avril.jpg",
        "Berville_L_CV_IA-avril.txt",
        model_name=model_name,
        normaliser=normaliser,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens
    )

    # Supposons qu'on maximise F1
    return metrics.get("f1", 0)

# --- Lancer Optuna ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best params:", study.best_params)
print("Best value (F1):", study.best_value)
print(study)