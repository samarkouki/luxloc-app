"""Reconvertit toutes les photos produits en vrai JPEG lisible par Pillow."""
import os
from PIL import Image
try:
    import pillow_avif
except ImportError:
    pass

dossier = "static/products"
ok, echecs = 0, []

for f in sorted(os.listdir(dossier)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue
    chemin = os.path.join(dossier, f)
    try:
        with Image.open(chemin) as im:
            im.load()
            img = im.convert("RGB")
        img.save(chemin, "JPEG", quality=85)
        print(f"OK  {f}")
        ok += 1
    except Exception as e:
        print(f"ECHEC {f} : {e}")
        echecs.append(f)

print(f"\n{ok} photos converties.")
if echecs:
    print(f"{len(echecs)} vraiment cassées (à re-télécharger) : {echecs}")
