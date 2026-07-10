#!/usr/bin/env python3
"""
LUXLOC — Téléchargeur automatique de photos produits
════════════════════════════════════════════════════

Ce script télécharge automatiquement les 18 photos des articles du seed
depuis Unsplash (gratuit, libre de droits) et les enregistre dans
`static/products/` avec les bons noms de fichier.

Une fois exécuté :
  1. Les 18 photos sont dans static/products/ avec le bon nommage
  2. Le seed enrichi (`POST /api/v1/reset-and-seed?confirm=yes`) les
     chargera automatiquement à la place des placeholders générés
  3. Elles apparaîtront dans le catalogue de l'app comme des articles
     réels mis en location

Prérequis :
  - Une clé API Unsplash gratuite : https://unsplash.com/oauth/applications
    → "New Application" → coche les conditions → tu récupères l'Access Key
  - `pip install requests` (déjà présent si tu as installé les deps du projet)

Utilisation :
  python download_product_photos.py YOUR_UNSPLASH_ACCESS_KEY

  ou avec variable d'environnement :
  set UNSPLASH_ACCESS_KEY=votre_cle   (Windows)
  export UNSPLASH_ACCESS_KEY=...      (Mac/Linux)
  python download_product_photos.py
"""
import os
import sys
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# LES 18 ARTICLES — nom du fichier + requêtes de recherche
# ═══════════════════════════════════════════════════════════════════════
# Chaque article a une requête primaire (idéale) et un fallback (plus large)
# au cas où Unsplash n'a rien avec la requête précise. On garde des angles
# et des cadrages neutres — le modèle IA ne dépend pas de l'angle exact.

ARTICLES = [
    ("01_lv_speedy.jpg",         "louis vuitton bag",       "brown monogram handbag"),
    ("02_gucci_marmont.jpg",     "gucci bag",               "black quilted leather handbag"),
    ("03_rolex_submariner.jpg",  "rolex submariner",        "luxury dive watch black"),
    ("04_hermes_kelly.jpg",      "hermes bag",              "tan leather luxury handbag"),
    ("05_vca_alhambra.jpg",      "gold pendant necklace",   "luxury gold jewelry"),
    ("06_burberry_trench.jpg",   "trench coat",             "beige trench coat fashion"),
    ("07_rayban_aviator.jpg",    "aviator sunglasses",      "gold frame sunglasses"),
    ("08_cartier_santos.jpg",    "cartier watch",           "luxury watch gold steel"),
    ("09_lv_neverfull.jpg",      "louis vuitton tote",      "brown checkered tote bag"),
    ("10_hermes_birkin.jpg",     "birkin bag",              "luxury tote handbag"),
    ("11_chanel_flap.jpg",       "chanel bag",              "black quilted chain bag"),
    ("12_rolex_daytona.jpg",     "rolex daytona",           "chronograph watch panda dial"),
    ("13_dior_lady.jpg",         "dior handbag",            "black designer handbag"),
    ("14_prada_galleria.jpg",    "prada bag",               "black saffiano leather bag"),
    ("15_rayban_wayfarer.jpg",   "wayfarer sunglasses",     "tortoise sunglasses"),
    ("16_cartier_love.jpg",      "gold bangle bracelet",    "cartier bracelet"),
    ("17_cartier_tank.jpg",      "cartier tank",            "rectangular vintage watch"),
    ("18_tommy_crossbody.jpg",   "tommy hilfiger",          "navy crossbody bag"),
]

OUTPUT_DIR = Path("static/products")
UNSPLASH_API = "https://api.unsplash.com/search/photos"


def search_unsplash(access_key, query, orientation="squarish"):
    """
    Interroge l'API Unsplash pour une requête donnée.
    Retourne l'URL directe de l'image « regular » (~1080px, largement suffisant)
    ou None si aucun résultat exploitable.
    """
    params = urllib.parse.urlencode({
        "query": query,
        "per_page": 5,
        "orientation": orientation,
        "content_filter": "high",  # évite les surprises visuelles
    })
    url = f"{UNSPLASH_API}?{params}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Client-ID {access_key}",
        "Accept-Version": "v1",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"    ✗ Clé API Unsplash invalide ou expirée")
            sys.exit(1)
        if e.code == 403:
            print(f"    ✗ Quota Unsplash dépassé (50/h en mode démo) — attends 1h")
            sys.exit(1)
        print(f"    ✗ Erreur HTTP {e.code} : {e.reason}")
        return None
    except Exception as e:
        print(f"    ✗ Erreur réseau : {e}")
        return None

    results = data.get("results", [])
    if not results:
        return None

    # On prend la première image dont l'auteur n'a pas refusé le tracking
    # (bonne pratique Unsplash) et de bonne qualité.
    for r in results:
        urls = r.get("urls", {})
        if urls.get("regular"):
            return urls["regular"]
    return None


def download_image(url, destination):
    """Télécharge une URL vers un fichier local."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "LUXLOC-photo-downloader/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(destination, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    ✗ Échec du téléchargement : {e}")
        return False


def get_access_key():
    """Récupère la clé API depuis l'argv ou l'environnement."""
    if len(sys.argv) > 1:
        return sys.argv[1].strip()
    key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    if not key:
        print(
            "Il me faut une clé API Unsplash.\n"
            "  → Va sur https://unsplash.com/oauth/applications\n"
            "  → Clique 'New Application', accepte les conditions\n"
            "  → Copie l'Access Key (~40 caractères)\n"
            "  → Relance ce script :  python download_product_photos.py TA_CLE\n"
        )
        sys.exit(1)
    return key


def main():
    access_key = get_access_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"════════════════════════════════════════════════")
    print(f"  Téléchargement de {len(ARTICLES)} photos LUXLOC")
    print(f"  Cible : {OUTPUT_DIR.resolve()}")
    print(f"════════════════════════════════════════════════\n")

    ok = 0
    skipped = 0
    failed = []

    for filename, primary_query, fallback_query in ARTICLES:
        dest = OUTPUT_DIR / filename

        if dest.exists():
            print(f"[= déjà là] {filename}")
            skipped += 1
            continue

        print(f"[…] {filename}  (\"{primary_query}\")")

        # 1re tentative : requête primaire
        url = search_unsplash(access_key, primary_query)
        # 2e tentative : requête plus large
        if not url:
            print(f"    ↻ retente avec \"{fallback_query}\"")
            url = search_unsplash(access_key, fallback_query)

        if not url:
            print(f"    ✗ aucun résultat exploitable")
            failed.append(filename)
            time.sleep(1)
            continue

        if download_image(url, dest):
            size_kb = dest.stat().st_size / 1024
            print(f"    ✓ {size_kb:.0f} KB")
            ok += 1
        else:
            failed.append(filename)

        # petit délai poli pour ne pas taper l'API trop vite
        time.sleep(0.6)

    # ─── Récapitulatif ───
    print(f"\n════════════════════════════════════════════════")
    print(f"  Résumé")
    print(f"════════════════════════════════════════════════")
    print(f"  ✓ Téléchargées : {ok}")
    print(f"  = Déjà présentes : {skipped}")
    if failed:
        print(f"  ✗ Échouées : {len(failed)}")
        for f in failed:
            print(f"    · {f}")
        print(f"\n  Pour les manquantes, télécharge-les à la main depuis")
        print(f"  https://unsplash.com et place-les dans {OUTPUT_DIR}/")
    print()
    print(f"  Prochaines étapes :")
    print(f"    1. git add static/products/")
    print(f"    2. git commit -m \"Photos reelles des articles du seed\"")
    print(f"    3. git push")
    print(f"    4. Sur Swagger, POST /api/v1/reset-and-seed?confirm=yes")
    print(f"════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
