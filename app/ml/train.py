"""
LUXLOC — Entraînement du modèle de détection de contrefaçon
═══════════════════════════════════════════════════════════

Ce script construit le jeu de données, entraîne un classifieur Random
Forest, l'évalue (train/test + validation croisée) et sauvegarde le
modèle ainsi que ses métriques.

─────────────────────────────────────────────────────────────────────
HONNÊTETÉ MÉTHODOLOGIQUE
Pour ce prototype, le jeu de données est généré de façon procédurale :
chaque image simule la signature visuelle d'un article authentique ou
d'une contrefaçon. Ces signatures reposent sur des différences réelles et
documentées :
  - un article authentique est généralement bien photographié (net) et
    présente un grain de matière structuré (cuir, métal travaillé) ;
  - une contrefaçon présente souvent soit une surface trop uniforme
    (matériau plastifié), soit une texture irrégulière, et des finitions
    moins régulières.
Les features sont, elles, extraites des pixels réels des images — c'est
un vrai pipeline de vision. Pour la version finale, il suffit de
remplacer la génération par le chargement d'un dataset réel (Kaggle,
scraping de Vestiaire Collective) : le reste du pipeline est inchangé.
─────────────────────────────────────────────────────────────────────

Usage : python -m app.ml.train
"""
import os
import json
import numpy as np
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from app.ml.features import extract_features, FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "counterfeit_rf.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

IMG_SIZE = 256
RNG = np.random.default_rng(42)


# ════════════════════════════════════════
# GÉNÉRATION DU JEU DE DONNÉES
# ════════════════════════════════════════

def _grain(scale, size=IMG_SIZE):
    """Génère un grain de matière en superposant du bruit à plusieurs échelles."""
    field = np.zeros((size, size), dtype=np.float64)
    octave = max(2, int(scale))
    weight = 1.0
    while octave <= size:
        small = RNG.random((octave, octave))
        up = np.asarray(Image.fromarray((small * 255).astype(np.uint8))
                        .resize((size, size), Image.BILINEAR), dtype=np.float64)
        field += weight * up
        octave *= 2
        weight *= 0.5
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return field


def make_authentic(size=IMG_SIZE):
    """
    Article authentique. La majorité sont bien photographiés (nets, grain
    structuré), mais ~25 % sont des cas difficiles : photo prise dans de
    mauvaises conditions (flou, faible contraste) — ce qui les rapproche
    visuellement d'une contrefaçon. Ce chevauchement est volontaire et
    réaliste : le modèle ne doit pas être parfait.
    """
    hard_case = RNG.random() < 0.25

    grain_scale = RNG.uniform(8, 26)
    base = _grain(grain_scale, size)

    if hard_case:
        # Authentique mal photographié
        contrast = RNG.uniform(0.45, 0.80)
        blur = RNG.uniform(0.8, 1.6)
    else:
        contrast = RNG.uniform(0.75, 1.15)
        blur = RNG.uniform(0.0, 0.7)

    img = 128 + (base - 0.5) * 255 * contrast

    # Coutures plutôt régulières
    spacing = RNG.integers(26, 42)
    for y in range(spacing, size, spacing):
        jitter = RNG.integers(-2, 3)
        yy = min(max(0, y + jitter), size - 2)
        img[yy:yy+1, :] += RNG.uniform(18, 42)

    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    if blur > 0.1:
        pil = pil.filter(ImageFilter.GaussianBlur(blur))
    return pil.convert("RGB")


def make_counterfeit(size=IMG_SIZE):
    """
    Contrefaçon. Trois profils :
      A) surface plastifiée / trop uniforme,
      B) texture chaotique et finitions imprécises,
      C) « super fake » bien réalisé — net, grain crédible : difficile à
         distinguer d'un authentique (~25 % des cas).
    """
    roll = RNG.random()

    if roll < 0.30:
        # Profil C — super fake (cas difficile)
        grain_scale = RNG.uniform(9, 24)
        base = _grain(grain_scale, size)
        contrast = RNG.uniform(0.70, 1.05)
        img = 128 + (base - 0.5) * 255 * contrast
        spacing = RNG.integers(26, 42)
        for y in range(spacing, size, spacing):
            jitter = RNG.integers(-3, 4)
            yy = min(max(0, y + jitter), size - 2)
            img[yy:yy+1, :] += RNG.uniform(14, 36)
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(RNG.uniform(0.0, 0.8)))

    elif roll < 0.65:
        # Profil A — plastifié
        grain_scale = RNG.uniform(34, 90)
        base = _grain(grain_scale, size)
        contrast = RNG.uniform(0.40, 0.78)
        img = 135 + (base - 0.5) * 255 * contrast
        spacing = RNG.integers(24, 46)
        for y in range(spacing, size, spacing):
            if RNG.random() < 0.7:
                jitter = RNG.integers(-3, 4)
                yy = min(max(0, y + jitter), size - 2)
                img[yy:yy+1, :] += RNG.uniform(8, 22)
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(RNG.uniform(0.3, 1.3)))

    else:
        # Profil B — texture chaotique
        fine = RNG.random((size, size))
        coarse = _grain(RNG.uniform(6, 16), size)
        mix = 0.55 * fine + 0.45 * coarse
        contrast = RNG.uniform(0.65, 1.05)
        img = 128 + (mix - 0.5) * 255 * contrast
        spacing = RNG.integers(22, 50)
        for y in range(spacing, size, spacing):
            jitter = RNG.integers(-6, 7)
            yy = min(max(0, y + jitter), size - 2)
            img[yy:yy+1, :] += RNG.uniform(10, 34)
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(RNG.uniform(0.0, 0.9)))

    return pil.convert("RGB")


def build_dataset(n_per_class=300):
    """Génère le dataset et extrait les features. Renvoie X, y."""
    X, y = [], []
    for _ in range(n_per_class):
        X.append(extract_features(make_authentic()))
        y.append(1)  # 1 = authentique
    for _ in range(n_per_class):
        X.append(extract_features(make_counterfeit()))
        y.append(0)  # 0 = contrefaçon
    return np.array(X), np.array(y)


# ════════════════════════════════════════
# ÉTUDE COMPARATIVE DE MODÈLES
# ════════════════════════════════════════

def comparative_study(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue plusieurs modèles sur le même split, génère une
    courbe ROC comparée, et renvoie les résultats triés par F1.
    Le Random Forest reste le modèle de production ; cette étude prouve
    que ce choix est justifié face à d'autres approches.
    """
    modeles = {
        "Random Forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=3, random_state=42, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(n_estimators=150, random_state=42)),
        "SVM (noyau RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=42)),
        "Réseau de neurones (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)),
        "Régression logistique": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)),
        "Référence naïve": make_pipeline(
            DummyClassifier(strategy="most_frequent")),
    }

    couleurs = {
        "Random Forest": "#C9A84C",
        "Gradient Boosting": "#6D2E46",
        "SVM (noyau RBF)": "#2D6A4F",
        "Réseau de neurones (MLP)": "#4A90D9",
        "Régression logistique": "#999999",
    }

    resultats = []
    plt.figure(figsize=(7, 6))

    for nom, clf in modeles.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        try:
            proba = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
            fpr, tpr, _ = roc_curve(y_test, proba)
            if nom != "Référence naïve":
                plt.plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})",
                         color=couleurs.get(nom, "#333"), linewidth=2)
        except Exception:
            auc = None

        resultats.append({
            "nom": nom,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "auc": round(auc, 4) if auc is not None else None,
        })

    # Ligne diagonale (classifieur aléatoire)
    plt.plot([0, 1], [0, 1], "--", color="#ccc", label="Aléatoire (AUC=0.500)")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC — comparaison des modèles")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    roc_path = os.path.join(MODEL_DIR, "roc_comparison.png")
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close()

    # Trie par F1 décroissant
    resultats.sort(key=lambda r: r["f1_score"], reverse=True)
    return resultats, roc_path


# ════════════════════════════════════════
# ENTRAÎNEMENT & ÉVALUATION
# ════════════════════════════════════════

def train():
    print("1/5 — Génération du jeu de données…")
    X, y = build_dataset(n_per_class=300)
    print(f"      {len(X)} images générées, {X.shape[1]} features chacune")

    print("2/5 — Séparation train / test (80 / 20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print("3/5 — Entraînement du Random Forest…")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("4/5 — Évaluation…")
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda t: t[1], reverse=True)

    # Étude comparative de modèles + courbe ROC
    print("      Étude comparative (6 modèles + courbe ROC)…")
    comparaison, roc_path = comparative_study(X_train, X_test, y_train, y_test)

    metrics = {
        "model": "RandomForestClassifier",
        "n_estimators": 200,
        "dataset_size": int(len(X)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features": FEATURE_NAMES,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": {
            "vrais_negatifs": int(cm[0, 0]),   # contrefaçon bien détectée
            "faux_positifs": int(cm[0, 1]),    # contrefaçon classée authentique
            "faux_negatifs": int(cm[1, 0]),    # authentique classé contrefaçon
            "vrais_positifs": int(cm[1, 1]),   # authentique bien détecté
        },
        "feature_importances": [
            {"feature": n, "importance": round(float(i), 4)}
            for n, i in importances
        ],
        "comparaison_modeles": comparaison,
    }

    print("5/5 — Sauvegarde du modèle et des métriques…")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 52)
    print("  RÉSULTATS DU MODÈLE")
    print("=" * 52)
    print(f"  Accuracy        : {acc:.1%}")
    print(f"  Precision       : {prec:.1%}")
    print(f"  Recall          : {rec:.1%}")
    print(f"  F1-score        : {f1:.1%}")
    print(f"  F1 (5-fold CV)  : {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    print("-" * 52)
    print("  Matrice de confusion (jeu de test) :")
    print(f"    Contrefaçons bien détectées : {cm[0,0]}")
    print(f"    Contrefaçons manquées       : {cm[0,1]}")
    print(f"    Authentiques mal classés    : {cm[1,0]}")
    print(f"    Authentiques bien détectés  : {cm[1,1]}")
    print("-" * 52)
    print("  Features les plus discriminantes :")
    for n, i in importances[:5]:
        print(f"    {n:<20} {i:.1%}")
    print("=" * 52)

    print("\n  ÉTUDE COMPARATIVE DE MODÈLES :")
    print(f"    {'Modèle':<28}{'Acc':>7}{'F1':>7}{'AUC':>7}")
    print("    " + "-" * 49)
    for r in comparaison:
        auc = f"{r['auc']:.3f}" if r['auc'] else "  —  "
        print(f"    {r['nom']:<28}{r['accuracy']:>7.1%}{r['f1_score']:>7.1%}{auc:>7}")
    print(f"\n  Courbe ROC sauvegardée : {roc_path}")
    print("=" * 52)
    print(f"\nModèle sauvegardé : {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train()
