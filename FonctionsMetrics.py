import os
# Calculs des distances
import editdistance
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Graphiques :
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(reference: str, predicted: str) -> dict:
    # Gestion des cas où les entrées sont des listes
    if isinstance(reference, list):
        reference = " ".join(reference)
    if isinstance(predicted, list):
        predicted = " ".join(predicted)

    # Mise en minuscules et tokenisation
    reference = reference.lower()
    predicted = predicted.lower()
    reference_words = word_tokenize(reference)
    predicted_words = word_tokenize(predicted)

    # CRR : Character Recognition Rate
    char_matches = sum(1 for a, b in zip(reference, predicted) if a == b)
    crr = char_matches / max(len(reference), 1)
    longeur = len(reference)

    # CER : Character Error Rate
    cer = edit_distance(reference, predicted) / max(len(reference), 1)

    # WRR : Word Recognition Rate
    word_matches = sum(1 for a, b in zip(reference_words, predicted_words) if a == b)
    wrr = word_matches / max(len(reference_words), 1)

    # WER : Word Error Rate
    wer = edit_distance(reference_words, predicted_words) / max(len(reference_words), 1)

    # Fuzzy ratio (Levenshtein à granularité texte)
    fuzzy_score = fuzz.ratio(reference, predicted)

    # BLEU score (n-grammes)
    bleu_score = sentence_bleu([reference_words], predicted_words) if predicted_words else 0

    # Précision / Rappel / F1 (approche set)
    ref_set = set(reference_words)
    pred_set = set(predicted_words)
    true_positive = len(ref_set & pred_set)
    false_positive = len(pred_set - ref_set)
    false_negative = len(ref_set - pred_set)

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Robustesse : simple version = 1 - CER
    robustness = 1 - cer

    return {
        'true_positive': round(true_positive, 3),
        'false_positive': round(false_positive, 3),
        'false_negative ': round(false_negative , 3),
        'CRR': round(crr, 3),
        'CER': round(cer, 3),
        'WRR': round(wrr, 3),
        'WER': round(wer, 3),
        'Fuzzy': round(fuzzy_score, 2),
        'BLEU': round(bleu_score, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-score': round(f1, 3),
        'Robustesse': round(robustness, 3),
        'longeur' :longeur
    }

# Les metriques en graphiques
def plot_ocr_metrics(df, output_folder='Gaphiques'):
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_to_plot = ['CRR', 'CER', 'F1-score', 'WER', 'Precision', 'Recall', 'BLEU', 'Robustesse']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='OCR_engine', y=metric, hue='Version', data=df)
        plt.title(f'Distribution de {metric} par OCR Engine et Version')
        plt.ylabel(metric)
        plt.xlabel('OCR Engine')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{metric}_boxplot.png")
        plt.close()

    print(f"Graphiques sauvegardés dans le dossier '{output_folder}'.")
