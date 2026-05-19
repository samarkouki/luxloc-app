"""
LUXLOC — Service de prédiction du modèle de détection de contrefaçon
═══════════════════════════════════════════════════════════════════

Charge le modèle Random Forest entraîné et fournit une prédiction pour
une image d'article. Utilisé par le pipeline de vérification IA à la
place du score heuristique précédent.

Le modèle renvoie une probabilité d'authenticité (0-1). Si le modèle
n'a pas encore été entraîné (fichier absent), le service bascule sur
le score heuristique de cv_analysis — l'application reste fonctionnelle.
"""
import os
import json
import joblib
from app.ml.features import extract_features, features_to_dict
from app.services import cv_analysis as cv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "counterfeit_rf.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Chargement paresseux du modèle (une seule fois)
_model = None
_metrics = None


def _load():
    global _model, _metrics
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, encoding="utf-8") as f:
                _metrics = json.load(f)
    return _model


def is_model_available():
    """Indique si le modèle entraîné est disponible."""
    return _load() is not None


def model_info():
    """Renvoie les métriques du modèle entraîné (ou None)."""
    _load()
    return _metrics


def predict_authenticity(img):
    """
    Prédit la probabilité qu'un article soit authentique à partir de sa photo.

    Renvoie un dict :
      - score        : probabilité d'authenticité 0-1
      - source       : 'modele_ml' ou 'heuristique' (repli)
      - features     : vecteur de features extrait
      - model_version: identifiant du modèle
    """
    model = _load()
    feats = extract_features(img)

    if model is not None:
        # classes_ = [0, 1] ; on prend la proba de la classe 1 (authentique)
        proba = model.predict_proba([feats])[0]
        idx = list(model.classes_).index(1)
        score = float(proba[idx])
        return {
            "score": round(score, 4),
            "source": "modele_ml",
            "model_version": "RandomForest-v1",
            "features": features_to_dict(feats),
        }

    # Repli heuristique si le modèle n'est pas entraîné
    return {
        "score": round(cv.global_quality_score(img), 4),
        "source": "heuristique",
        "model_version": "fallback",
        "features": features_to_dict(feats),
    }
