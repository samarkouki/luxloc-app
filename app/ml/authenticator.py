"""
LUXLOC — Authentification par EfficientNet-B0 (luxury_authenticator_v1)
════════════════════════════════════════════════════════════════════

Remplace le Random Forest (app.ml.predict) comme source du score
d'authenticité à l'étape 2 du pipeline de vérification (voir
app/routers/verification.py). Réutilise la classe Predictor de
luxury_authenticator_v1/inference/predict.py sans la modifier.

Chargement PARESSEUX : le modèle n'est chargé qu'à la première
vérification appelant is_available(), jamais à l'import de ce module ni
au démarrage de FastAPI — le code d'origine (luxury_authenticator_v1/
app/main.py) instancie Predictor au niveau du module, ce qui chargerait
torch et 16 Mo de poids à chaque démarrage de l'API, y compris quand
personne ne vérifie d'article. On ne reproduit pas ça.

Si le chargement échoue pour quelque raison que ce soit (fichier .pth
absent, torch/torchvision non installés, mémoire insuffisante...),
is_available() renvoie False et n'est plus retenté ensuite (évite de
retenter à chaque requête un chargement durablement impossible) ; le
pipeline de vérification bascule alors silencieusement sur le Random
Forest (app.ml.predict), qui a lui-même un repli heuristique. L'API
reste pleinement fonctionnelle dans tous les cas.
"""
import os
import json

MODEL_VERSION = "EfficientNet-B0-v1"

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AUTHENTICATOR_DIR = os.path.join(_APP_ROOT, "luxury_authenticator_v1")
MODEL_PATH = os.path.join(_AUTHENTICATOR_DIR, "models", "luxury_authenticator.pth")
METRICS_PATH = os.path.join(_AUTHENTICATOR_DIR, "models", "metrics.json")

_predictor = None
_load_attempted = False
_load_error = None


def _load():
    global _predictor, _load_attempted, _load_error
    if _load_attempted:
        return _predictor
    _load_attempted = True
    try:
        from luxury_authenticator_v1.inference.predict import Predictor
        _predictor = Predictor(MODEL_PATH)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        _predictor = None
    return _predictor


def is_available():
    """Indique si EfficientNet a pu être chargé (True au plus une tentative)."""
    return _load() is not None


def status():
    """
    État du chargement, sans en déclencher un : "non_sollicite" (aucune
    vérification n'a encore eu lieu), "actif" (chargé avec succès) ou
    "indisponible" (tenté puis échoué — voir load_error()). Utilisé par
    /api/v1/health pour ne pas provoquer de chargement juste en consultant
    l'état du service.
    """
    if not _load_attempted:
        return "non_sollicite"
    return "actif" if _predictor is not None else "indisponible"


def load_error():
    """Raison du dernier échec de chargement, pour diagnostic (health, logs)."""
    return _load_error


def model_info():
    """Renvoie le contenu de metrics.json (ou None si absent)."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict_one(img):
    """
    Analyse une image et renvoie le score d'authenticité (probabilité
    'real') ainsi que la confiance du modèle. Suppose le modèle déjà
    chargé — appeler is_available() avant.

    Une confiance sous 0,80 (seuil du modèle d'origine) est considérée
    non concluante : la photo ne doit pas compter comme preuve
    d'authenticité, à charge de l'appelant de l'exclure du calcul.
    """
    predictor = _load()
    result = predictor.predict(img)
    return {
        "score": round(result["real_probability"], 4),
        "confidence": round(result["confidence"], 4),
        "concluant": result["confidence"] >= 0.80,
        "verdict": result["verdict"],
    }
