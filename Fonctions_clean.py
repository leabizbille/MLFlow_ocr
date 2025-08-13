import unicodedata
import re

def normalize_text(text: str) -> str:
    # Étape 1 : Gestion du cas où l'entrée est une liste de chaînes
    if isinstance(text, list):
        # On concatène tous les éléments de la liste en une seule chaîne
        text = " ".join(text)
    # Étape 2 : Normalisation Unicode
    # NFKD décompose les caractères accentués en caractères de base + diacritiques
    text = unicodedata.normalize('NFKD', text)
    # Étape 3 : Suppression des accents et caractères non-ASCII
    # encode en ASCII en ignorant les caractères non-ASCII, puis decode en UTF-8
    # Exemple : 'é' -> 'e'
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # Étape 4 : Supprimer les espaces multiples
    # remplace toutes les suites d'espaces par un seul espace
    text = re.sub(r'\s+', ' ', text)
    # Étape 5 : Supprimer la ponctuation
    # garde uniquement les lettres, chiffres et espaces
    text = re.sub(r'[^\w\s]', '', text)
    # Étape 6 : Conversion en minuscules et suppression des espaces en début/fin
    text = text.lower().strip()
    # Étape 7 : Debug / affichage intermédiaire (optionnel)
    print(text)
    return text
