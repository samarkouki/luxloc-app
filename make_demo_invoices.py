"""Génère 2 factures de démo : authentique + falsifiée."""
import base64
from main import generate_invoice_b64

def sauver(b64, nom):
    raw = base64.b64decode(b64.split(",")[1])
    open(nom, "wb").write(raw)
    print("Créée :", nom)

# Authentique — prix cohérent 1250 €
auth = generate_invoice_b64("Louis Vuitton", "Speedy 25", "bags", 1250,
                            "LV2024-08742", "14 / 03 / 2024", tampered=False)
sauver(auth, "facture_authentique.jpg")

# Falsifiée — montant retouché, prix gonflé à 6900 €
faux = generate_invoice_b64("Louis Vuitton", "Speedy 25", "bags", 6900,
                            "LV2024-08742", "14 / 03 / 2024", tampered=True)
sauver(faux, "facture_falsifiee.jpg")
print("\nPretes pour la demo.")
