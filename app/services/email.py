"""
Envoi d'e-mails transactionnels via l'API HTTP de Resend (https://resend.com).

Aucune dépendance ajoutée : un simple appel HTTPS avec la bibliothèque
standard. Configuration par variables d'environnement (sur Railway :
onglet Variables du service) :
  RESEND_API_KEY  clé API Resend (obligatoire pour envoyer)
  EMAIL_FROM      expéditeur, ex. "LUXLOC <contact@mondomaine.fr>". Sans
                  domaine vérifié chez Resend, l'adresse de test
                  onboarding@resend.dev n'envoie qu'à l'adresse du compte Resend.
"""
import json
import os
import urllib.request

RESEND_URL = "https://api.resend.com/emails"


def is_configured():
    return bool(os.getenv("RESEND_API_KEY"))


def send_email(to, subject, html):
    """Envoie un e-mail. Lève une exception si l'envoi échoue."""
    payload = json.dumps({
        "from": os.getenv("EMAIL_FROM", "LUXLOC <onboarding@resend.dev>"),
        "to": [to],
        "subject": subject,
        "html": html,
    }).encode("utf-8")
    req = urllib.request.Request(RESEND_URL, data=payload, method="POST", headers={
        "Authorization": f"Bearer {os.getenv('RESEND_API_KEY')}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status
