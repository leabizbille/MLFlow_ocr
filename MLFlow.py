import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Exemple avec un modèle seq2seq pour correction
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
ocr_correction_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Suivi avec MLflow
mlflow.start_run()
mlflow.log_param("model_name", "bart-base")
mlflow.log_param("ocr_correction", True)

texte_ocr = "Exemple de text avec errurs"
texte_corrige = ocr_correction_pipeline(texte_ocr)[0]['generated_text']

mlflow.log_metric("text_length_before", len(texte_ocr))
mlflow.log_metric("text_length_after", len(texte_corrige))

mlflow.end_run()
