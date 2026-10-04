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

LIBÉRATION APRÈS INACTIVITÉ : un thread démon libère la référence au
modèle après IDLE_UNLOAD_SECONDS sans utilisation (voir unload_if_idle).
Ça ne coûte rien aux appels normaux (une simple mise à jour d'horodatage),
et la vérification suivante recharge le modèle en ~0,3s (torch reste
importé dans le process — mesuré, à comparer aux ~7-9s du tout premier
chargement) : aucun coût perceptible pour une démonstration où les
vérifications sont rapprochées.

Limite mesurée, à ne pas survendre : sur cette machine de test (Windows),
libérer la référence et forcer gc.collect() ne fait PAS redescendre la
mémoire résidente (RSS) du process — l'allocateur mémoire garde les pages
libérées pour les réutiliser plutôt que de les rendre à l'OS. Le cycle
charge/libère/recharge est sans fuite (RSS stable sur plusieurs cycles,
pas de croissance non bornée), mais ne doit pas être présenté comme une
façon de faire baisser la consommation mémoire observée en production —
seul un redémarrage du process la fait redescendre. Le comportement de
l'allocateur glibc sur Railway (Linux) peut différer ; à vérifier après
déploiement plutôt qu'à supposer.
"""
import os
import json
import gc
import threading
import time

MODEL_VERSION = "EfficientNet-B0-v1"

# Durée d'inactivité avant libération du modèle. 30 minutes : pour une
# démonstration, mieux vaut garder le modèle chaud (voir la limite mesurée
# plus haut — la libération ne fait de toute façon pas redescendre la
# mémoire observée) qu'imposer un rechargement à la moindre pause.
IDLE_UNLOAD_SECONDS = 1800

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AUTHENTICATOR_DIR = os.path.join(_APP_ROOT, "luxury_authenticator_v1")
MODEL_PATH = os.path.join(_AUTHENTICATOR_DIR, "models", "luxury_authenticator.pth")
METRICS_PATH = os.path.join(_AUTHENTICATOR_DIR, "models", "metrics.json")

_predictor = None
_load_attempted = False
_load_error = None
_last_used = None
_last_unloaded_at = None
_lock = threading.Lock()
_watcher_started = False


def _touch():
    global _last_used
    _last_used = time.monotonic()


def _start_idle_watcher():
    """
    Démarre (une seule fois par process) le thread démon qui surveille
    l'inactivité. Lancé uniquement après un premier chargement réussi —
    inutile de faire tourner ce thread si le modèle n'a jamais servi ou
    est durablement indisponible.
    """
    global _watcher_started
    if _watcher_started:
        return
    _watcher_started = True

    def _loop():
        while True:
            time.sleep(60)
            with _lock:
                if _predictor is not None and _last_used is not None:
                    if time.monotonic() - _last_used > IDLE_UNLOAD_SECONDS:
                        _unload_locked()

    threading.Thread(target=_loop, name="efficientnet-idle-unload", daemon=True).start()


def _unload_locked():
    """Libère le modèle. À appeler avec _lock déjà acquis."""
    global _predictor, _load_attempted, _last_used, _last_unloaded_at
    _predictor = None
    _load_attempted = False  # autorise un rechargement complet au prochain appel
    _last_used = None
    _last_unloaded_at = time.monotonic()
    gc.collect()


def unload_if_idle():
    """
    Point d'entrée testable de la libération par inactivité (le thread
    démon appelle la même logique). Renvoie True si une libération a eu
    lieu.
    """
    with _lock:
        if _predictor is not None and _last_used is not None:
            if time.monotonic() - _last_used > IDLE_UNLOAD_SECONDS:
                _unload_locked()
                return True
    return False


def cpu_quota():
    """Nombre de processeurs réellement alloués par le conteneur (cgroup v2), ou None."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        return None if quota == "max" else round(int(quota) / int(period), 2)
    except (OSError, ValueError):
        return None


def torch_threads():
    """
    Threads de calcul pour torch : TORCH_THREADS si défini, sinon le quota
    du conteneur arrondi à l'inférieur, plafonné à 2 (au-delà, le gain est
    faible pour une image à la fois, et le risque de bridage remonte).
    """
    env = os.getenv("TORCH_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    quota = cpu_quota()
    return max(1, min(2, int(quota))) if quota else 2


def _load():
    global _predictor, _load_attempted, _load_error
    with _lock:
        if _load_attempted:
            if _predictor is not None:
                _touch()
            return _predictor
        _load_attempted = True
        try:
            import torch
            # Par défaut torch lance autant de threads que de processeurs
            # visibles. Dans un conteneur Railway, ce sont ceux de la machine
            # hôte, alors que le quota CPU n'en autorise qu'une fraction : les
            # threads épuisent le quota et le noyau bride le process. Mesuré en
            # production : 0,5 s pour 4 photos en temps normal, 35 à 37 s une
            # fois bridé. On limite donc le nombre de threads.
            torch.set_num_threads(torch_threads())
            torch.set_num_interop_threads(1)
            from luxury_authenticator_v1.inference.predict import Predictor
            _predictor = Predictor(MODEL_PATH)
            _touch()
            _start_idle_watcher()
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
    vérification n'a encore eu lieu), "actif" (chargé avec succès),
    "indisponible" (tenté puis échoué — voir load_error()), ou
    "libere_inactivite" (chargé avec succès par le passé, puis libéré
    après IDLE_UNLOAD_SECONDS sans utilisation — se recharge tout seul à
    la prochaine vérification). Utilisé par /api/v1/health pour ne pas
    provoquer de chargement juste en consultant l'état du service.
    """
    if not _load_attempted:
        return "libere_inactivite" if _last_unloaded_at is not None else "non_sollicite"
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
