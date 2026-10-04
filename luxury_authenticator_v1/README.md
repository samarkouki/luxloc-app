# Luxury Authenticator — MVP complet

Pipeline : **dataset → téléchargement → preprocessing → entraînement → évaluation → modèle `.pth` → API FastAPI → interface web → upload photo → résultat**.

## Ce que fait ce MVP
- Fine-tune **EfficientNet-B0 pré-entraîné sur ImageNet**.
- Classification binaire `real` / `fake`.
- Split stratifié 70/15/15.
- Augmentation d’images pendant l’entraînement.
- Pondération de la loss pour limiter le biais lié au déséquilibre de classes.
- Support natif **Apple Silicon / MPS** lorsque disponible.
- Évaluation : accuracy, précision, recall, F1, ROC-AUC, matrice de confusion.
- API FastAPI `POST /predict`.
- Interface web locale pour déposer une photo.

## 0. Prérequis
Python 3.11 recommandé.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Télécharger le dataset

Le script utilise par défaut un dataset Kaggle public connu sous le nom **Fake Brand Data** :

```bash
python data/download_dataset.py
```

Si Kaggle te demande une authentification, crée une clé API dans ton compte Kaggle et place `kaggle.json` dans `~/.kaggle/`, puis relance la commande.

Tu peux aussi fournir un autre slug Kaggle :

```bash
python data/download_dataset.py --dataset USER/DATASET
```

## 2. Préparer les données

Le script détecte automatiquement les dossiers/fichiers contenant des mots comme `real`, `genuine`, `authentic`, `fake`, `counterfeit`, etc.

```bash
python data/prepare.py
```

Résultat :

```text
data/processed/
  train/real/
  train/fake/
  val/real/
  val/fake/
  test/real/
  test/fake/
```

## 3. Entraîner

Pour un premier run rapide :

```bash
python training/train.py --epochs 8 --batch-size 32
```

Le meilleur checkpoint de validation est sauvegardé dans :

```text
models/luxury_authenticator.pth
```

Sur Mac M2, PyTorch essaiera d’utiliser **MPS** automatiquement.

## 4. Évaluer

```bash
python training/evaluate.py
```

Les métriques sont écrites dans :

```text
models/metrics.json
```

## 5. Tester une photo depuis le terminal

```bash
python inference/predict.py /chemin/vers/photo.jpg
```

Exemple de sortie :

```json
{
  "status": "AUTHENTIQUE PROBABLE",
  "fake_probability": 0.08,
  "authentic_probability": 0.92,
  "confidence": 0.92
}
```

## 6. Lancer l'API + interface web

```bash
uvicorn app.main:app --reload
```

Puis ouvre : `http://127.0.0.1:8000`

Documentation API : `http://127.0.0.1:8000/docs`

## Architecture

```text
Photo
  ↓
Resize / normalization / augmentation (train)
  ↓
EfficientNet-B0 pré-entraîné
  ↓
2 classes : REAL / FAKE
  ↓
Probabilités
  ↓
FastAPI
  ↓
Interface Web
```

## IMPORTANT pour une vraie solution d'authentification

Ce MVP est une **preuve de concept**. Une photo générique d’un sac ne suffit pas à garantir l’authenticité. Les travaux académiques sur l’authentification de sacs de luxe utilisent notamment des images de parties discriminantes telles que le code, l’embossage, la face, le logo, l’étiquette, la serrure et la fermeture, avec plusieurs images par article. Un dataset académique cité dans la littérature contient 74 916 images Chanel, Gucci, Louis Vuitton et Prada, annotées real/counterfeit et séparées en train/validation/test. Le projet pourra être renforcé avec une stratégie multi-vues : plusieurs photos, crop de détails, score par zone et agrégation finale.

Ne présente donc pas la sortie comme une certification officielle.

## Phase 2 recommandée

1. Détecter automatiquement les zones importantes (logo, label, code, stitching, zip...).
2. Encoder chaque crop avec un modèle vision.
3. Comparer aux références de modèles authentiques.
4. Fusionner les scores des différents détails.
5. Ajouter un statut `INSUFFICIENT_EVIDENCE` lorsque la photo ne montre pas les détails nécessaires.
