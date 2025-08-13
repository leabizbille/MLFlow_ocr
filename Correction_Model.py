import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def corriger_ocr_avec_mlflow(texte_ocr: str):
    """
    Corrige un texte OCR avec le modèle BART et enregistre les infos dans MLflow.
    
    Args:
        texte_ocr (str): texte issu de l'OCR.
    
    Returns:
        dict: texte original, texte corrigé et métriques associées.
    """
    # --- Initialisation modèle ---
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
    ocr_correction_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    # --- Début du suivi MLflow ---
    mlflow.start_run()
    mlflow.log_param("model_name", "facebook/bart-base")
    mlflow.log_param("ocr_correction", True)

    # --- Correction ---
    texte_corrige = ocr_correction_pipeline(texte_ocr)[0]['generated_text']

    # --- Logging des métriques ---
    mlflow.log_metric("text_length_before", len(texte_ocr))
    mlflow.log_metric("text_length_after", len(texte_corrige))

    # --- Fin du run ---
    mlflow.end_run()

    return {
        "texte_original": texte_ocr,
        "texte_corrige": texte_corrige,
        "len_before": len(texte_ocr),
        "len_after": len(texte_corrige)
    }


resultat = corriger_ocr_avec_mlflow("Exemle de text avec erurs")
print(resultat)
