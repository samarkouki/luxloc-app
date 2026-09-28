"""
LUXLOC — Application Intégrée (Backend + Frontend)
Lance avec : uvicorn main:app --reload
Ouvre : http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import text, inspect
from app.routers import articles, users, verification, transactions, returns, admin
from app.database import engine, Base, SessionLocal
from app.models.models import *
import os
import hashlib
import base64
import io
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
try:
    import pillow_avif
except ImportError:
    pass

Base.metadata.create_all(bind=engine)


# ════════════════════════════════════════════════════════════════════
# MIGRATION LÉGÈRE — nouvelles valeurs d'ENUM Postgres
# ════════════════════════════════════════════════════════════════════
# Base.metadata.create_all() crée les types ENUM Postgres s'ils n'existent
# pas encore, mais ne modifie JAMAIS un type déjà présent. Si on ajoute une
# valeur à un Enum Python (ex: UserRole.ADMIN) après le premier déploiement,
# la base reste bloquée sur l'ancien type — toute insertion avec la
# nouvelle valeur échoue et annule la transaction en cours. On répare ça
# ici au démarrage, de façon idempotente et sans effet sur SQLite (où les
# enums sont de simples chaînes, sans contrainte de type dédiée).
if engine.dialect.name == "postgresql":
    with engine.connect() as _conn:
        _conn = _conn.execution_options(isolation_level="AUTOCOMMIT")
        for _value in UserRole:
            # SQLAlchemy persiste le NOM du membre de l'enum Python (ex.
            # "ADMIN"), pas sa valeur (ex. "admin") — c'est bien .name qu'il
            # faut ajouter au type Postgres, pas .value.
            try:
                _conn.execute(text(
                    f"ALTER TYPE userrole ADD VALUE IF NOT EXISTS '{_value.name}'"))
            except Exception as _e:
                print(f"[migration] userrole '{_value.name}' : {_e}")


# ════════════════════════════════════════════════════════════════════
# MIGRATION LÉGÈRE — nouvelles colonnes sur des tables existantes
# ════════════════════════════════════════════════════════════════════
# Même limitation que ci-dessus : Base.metadata.create_all() crée les
# tables qui n'existent pas encore, mais ne modifie JAMAIS une table déjà
# présente. Sur une base neuve, create_all a déjà créé ces colonnes et la
# vérification ci-dessous est un no-op. Sur une base de production
# existante (le cas Railway/Postgres), c'est cette étape qui les ajoute
# réellement, sans jamais toucher aux données déjà en place.
# On vérifie nous-mêmes la présence de la colonne plutôt que de compter
# sur "ADD COLUMN IF NOT EXISTS" (absent des anciennes versions de
# SQLite) : cette approche fonctionne à l'identique sur les deux moteurs.
def _ensure_column(table, column, ddl_type):
    try:
        existing = {c["name"] for c in inspect(engine).get_columns(table)}
    except Exception as _e:
        print(f"[migration] impossible d'inspecter {table} : {_e}")
        return
    if column in existing:
        return
    try:
        with engine.connect() as _conn:
            _conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
            _conn.commit()
        print(f"[migration] colonne ajoutée : {table}.{column}")
    except Exception as _e:
        print(f"[migration] échec ajout {table}.{column} : {_e}")

_ensure_column("users", "is_active", "BOOLEAN DEFAULT TRUE")
_ensure_column("articles", "is_archived", "BOOLEAN DEFAULT FALSE")
_ensure_column("articles", "archived_at", "TIMESTAMP")


# ════════════════════════════════════════════════════════════════════
# CHEMINS ABSOLUS — indépendants du répertoire courant du process
# ════════════════════════════════════════════════════════════════════
# Railway ne garantit pas que uvicorn soit lancé depuis la racine du
# projet. Un chemin relatif ("static/products") échouerait alors en
# silence et on retomberait sur les placeholders sans aucun message.
# On ancre donc tous les chemins sur l'emplacement de ce fichier.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PRODUCT_PHOTOS_DIR = os.path.join(STATIC_DIR, "products")


# ════════════════════════════════════════════════════════════════════
# GÉNÉRATEURS DE DONNÉES DE DÉMO — factures et photos d'articles
# ════════════════════════════════════════════════════════════════════

BOUTIQUE_ADDRESSES = {
    "Louis Vuitton": "Boutique Louis Vuitton — 101 av. des Champs-Élysées, 75008 Paris",
    "Hermès": "Hermès Paris — 24 rue du Faubourg Saint-Honoré, 75008 Paris",
    "Chanel": "CHANEL — 31 rue Cambon, 75001 Paris",
    "Gucci": "Gucci Boutique — 2 rue du Faubourg Saint-Honoré, 75008 Paris",
    "Rolex": "Rolex Boutique — 8 place Vendôme, 75001 Paris",
    "Cartier": "Cartier — 13 rue de la Paix, 75002 Paris",
    "Van Cleef & Arpels": "Van Cleef & Arpels — 22 place Vendôme, 75001 Paris",
    "Burberry": "Burberry Boutique — 8 boulevard Malesherbes, 75008 Paris",
    "Dior": "Dior Boutique — 30 avenue Montaigne, 75008 Paris",
    "Prada": "Prada Paris — 6 rue du Faubourg Saint-Honoré, 75008 Paris",
    "Ray-Ban": "Optique du Faubourg — 56 rue du Faubourg Saint-Honoré, 75008 Paris",
    "Tommy Hilfiger": "Tommy Hilfiger Store — 65 av. des Champs-Élysées, 75008 Paris",
}


def _font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def generate_invoice_b64(brand, model, category, retail_price, invoice_num, date_str,
                         tampered=False):
    """
    Génère une facture de boutique réaliste (en-tête, cadre, tableau,
    filigrane, cachet PAYÉ). Si tampered=True, le montant TOTAL est
    grossièrement retouché (réécrit par-dessus) pour la démo de détection
    de falsification.
    """
    W, H = 850, 1150
    CREAM = (250, 248, 244)
    INK = (26, 22, 20)
    GOLD = (201, 168, 76)
    GREY = (120, 116, 110)
    LIGHT = (232, 228, 220)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Cadre extérieur doré
    d.rectangle([25, 25, W - 25, H - 25], outline=GOLD, width=2)
    d.rectangle([32, 32, W - 32, H - 32], outline=LIGHT, width=1)

    # Filigrane diagonal (nom de la marque répété, très clair)
    wm = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wm)
    for row in range(-2, 14):
        for col in range(-2, 6):
            wd.text((col * 200 - 60, row * 95 + 40), brand.upper(),
                    fill=(201, 168, 76, 14), font=_font(26, bold=True))
    wm = wm.rotate(30, expand=False)
    img.paste(Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img)

    # ── EN-TÊTE : nom de la maison centré ──
    brand_font = _font(46, bold=True)
    bbox = d.textbbox((0, 0), brand.upper(), font=brand_font)
    bw = bbox[2] - bbox[0]
    d.text(((W - bw) / 2, 70), brand.upper(), fill=INK, font=brand_font)
    # Sous-titre "MAISON DE LUXE"
    sub = "M A I S O N   D E   L U X E"
    sbbox = d.textbbox((0, 0), sub, font=_font(12))
    d.text(((W - (sbbox[2] - sbbox[0])) / 2, 128), sub, fill=GOLD, font=_font(12))

    # Filet doré séparateur
    d.rectangle([70, 165, W - 70, 167], fill=GOLD)

    # ── Coordonnées boutique (gauche) ──
    d.text((70, 190), BOUTIQUE_ADDRESSES.get(brand, "Boutique officielle — Paris"),
           fill=GREY, font=_font(12))
    d.text((70, 210), "SIRET 552 084 793 00012  ·  TVA FR 12 552084793",
           fill=GREY, font=_font(11))
    d.text((70, 228), "Tél. +33 1 53 57 52 00  ·  contact@maison-luxe.fr",
           fill=GREY, font=_font(11))

    # ── Bloc FACTURE encadré (droite) ──
    d.rectangle([W - 300, 255, W - 70, 340], outline=GOLD, width=1, fill=CREAM)
    d.text((W - 285, 268), "FACTURE", fill=INK, font=_font(16, bold=True))
    d.text((W - 285, 296), f"N°  {invoice_num}", fill=INK, font=_font(13))
    d.text((W - 285, 316), f"Date  {date_str}", fill=GREY, font=_font(12))

    # ── Infos client / conseiller (gauche) ──
    d.text((70, 268), "FACTURÉ À", fill=GOLD, font=_font(10))
    d.text((70, 286), "Client privilégié", fill=INK, font=_font(13, bold=True))
    d.text((70, 306), "Conseiller : M. Laurent D.", fill=GREY, font=_font(12))

    # ── TABLEAU de désignation ──
    ty = 390
    d.rectangle([70, ty, W - 70, ty + 34], fill=INK)
    d.text((85, ty + 9), "DÉSIGNATION", fill="white", font=_font(12, bold=True))
    d.text((W - 250, ty + 9), "QTÉ", fill="white", font=_font(12, bold=True))
    d.text((W - 180, ty + 9), "MONTANT", fill="white", font=_font(12, bold=True))

    cat_label = {"bags": "Maroquinerie", "watches": "Horlogerie",
                 "jewelry": "Joaillerie", "clothing": "Prêt-à-porter",
                 "eyewear": "Lunetterie"}.get(category, "Article")

    ry = ty + 50
    d.text((85, ry), f"{brand} — {model}", fill=INK, font=_font(14, bold=True))
    d.text((85, ry + 22), f"{cat_label} · Réf. {invoice_num}", fill=GREY, font=_font(11))
    d.text((85, ry + 40), "Livré avec dustbag, boîte et certificat d'authenticité",
           fill=GREY, font=_font(11))
    d.text((W - 245, ry), "1", fill=INK, font=_font(14))

    def fmt(v):
        return f"{v:,.2f}".replace(",", " ").replace(".", ",")

    d.text((W - 180, ry), f"{fmt(retail_price)} €", fill=INK, font=_font(14))

    # Ligne de séparation
    d.line([70, ry + 75, W - 70, ry + 75], fill=LIGHT, width=1)

    # ── Récapitulatif HT / TVA / TTC (droite) ──
    ht = round(retail_price / 1.20, 2)
    tva = round(retail_price - ht, 2)
    sy = ry + 95
    d.text((W - 320, sy), "Sous-total HT", fill=GREY, font=_font(13))
    d.text((W - 180, sy), f"{fmt(ht)} €", fill=INK, font=_font(13))
    d.text((W - 320, sy + 26), "TVA 20 %", fill=GREY, font=_font(13))
    d.text((W - 180, sy + 26), f"{fmt(tva)} €", fill=INK, font=_font(13))

    # ── Bloc TOTAL TTC mis en valeur ──
    total_y = sy + 60
    d.rectangle([W - 340, total_y, W - 70, total_y + 46], fill=GOLD)
    d.text((W - 325, total_y + 13), "TOTAL TTC", fill=INK, font=_font(16, bold=True))

    # >>> Le montant total — retouché si tampered <
    if tampered:
        # Simulation d'une retouche grossière : rectangle blanc collé
        # par-dessus l'ancien montant, puis nouveau montant réécrit.
        # Cela crée une rupture de compression (détectable par ELA) et
        # un montant incohérent (détectable par le veto prix OCR).
        d.rectangle([W - 205, total_y + 8, W - 78, total_y + 40], fill=(255, 255, 255))
        d.text((W - 200, total_y + 12), f"{fmt(retail_price)} €",
               fill=(10, 10, 10), font=_font(17, bold=True))
    else:
        d.text((W - 200, total_y + 13), f"{fmt(retail_price)} €",
               fill=INK, font=_font(16, bold=True))

    # ── Cachet "PAYÉ" incliné ──
    stamp = Image.new("RGBA", (240, 90), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.rectangle([4, 4, 236, 86], outline=(46, 106, 79, 180), width=3)
    sd.text((30, 26), "P A Y É", fill=(46, 106, 79, 180), font=_font(34, bold=True))
    stamp = stamp.rotate(-12, expand=True)
    img.paste(stamp, (95, total_y + 10), stamp)

    # ── Pied de page ──
    fy = H - 180
    d.line([70, fy, W - 70, fy], fill=LIGHT, width=1)
    d.text((70, fy + 15), f"Article couvert par la garantie {brand} (2 ans).",
           fill=GREY, font=_font(11))
    d.text((70, fy + 33), "Cette facture vaut preuve d'achat et de propriété.",
           fill=GREY, font=_font(11))
    d.text((70, fy + 51), "Règlement comptant — Carte bancaire ····  ····  ····  4417",
           fill=GREY, font=_font(11))
    d.text((70, fy + 85), "Merci de votre confiance.", fill=GOLD, font=_font(13, bold=True))
    sig = f"— {brand} · Paris —"
    sbb = d.textbbox((0, 0), sig, font=_font(11))
    d.text(((W - (sbb[2] - sbb[0])) / 2, H - 65), sig, fill=(180, 176, 170),
           font=_font(11))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_product_photo_b64(brand, model):
    palettes = {
        "Louis Vuitton": [(61, 43, 31), (139, 105, 60)],
        "Hermès": [(199, 107, 42), (245, 213, 160)],
        "Chanel": [(26, 26, 26), (201, 168, 76)],
        "Gucci": [(26, 60, 52), (201, 168, 76)],
        "Rolex": [(10, 61, 42), (201, 168, 76)],
        "Cartier": [(90, 10, 10), (201, 168, 76)],
        "Van Cleef & Arpels": [(44, 22, 84), (201, 168, 76)],
        "Burberry": [(139, 105, 20), (232, 213, 160)],
        "Dior": [(232, 224, 208), (139, 125, 94)],
        "Prada": [(26, 26, 26), (74, 74, 74)],
        "Ray-Ban": [(26, 26, 46), (15, 52, 96)],
        "Tommy Hilfiger": [(0, 31, 92), (200, 16, 46)],
    }
    c1, c2 = palettes.get(brand, [(60, 60, 60), (180, 150, 90)])
    img = Image.new("RGB", (600, 600), c1)
    d = ImageDraw.Draw(img)
    for i in range(600):
        ratio = i / 600
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        d.line([(i, 0), (0, i)], fill=(r, g, b))
    overlay = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.ellipse([100, 100, 500, 500], fill=(0, 0, 0, 50))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    monogram = (brand[:2] if len(brand) > 1 else brand[0]).upper()
    if brand == "Louis Vuitton": monogram = "LV"
    elif brand == "Chanel": monogram = "CC"
    elif brand == "Hermès": monogram = "H"
    elif brand == "Gucci": monogram = "GG"
    elif brand == "Van Cleef & Arpels": monogram = "VCA"
    elif brand == "Tommy Hilfiger": monogram = "TH"
    d.text((300, 270), monogram, fill=(255, 255, 255, 200),
           font=_font(180, bold=True), anchor="mm")
    d.text((300, 410), brand.upper(), fill=(255, 255, 255, 180),
           font=_font(20, bold=True), anchor="mm")
    d.text((300, 445), model.upper(), fill=(220, 220, 220),
           font=_font(13), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


PHOTO_FILES = [
    "01_lv_speedy.jpg", "02_gucci_marmont.jpg", "03_rolex_submariner.jpg",
    "04_hermes_kelly.jpg", "05_vca_alhambra.jpg", "06_burberry_trench.jpg",
    "07_rayban_aviator.jpg", "08_cartier_santos.jpg", "09_lv_neverfull.jpg",
    "10_hermes_birkin.jpg", "11_chanel_flap.jpg", "12_rolex_daytona.jpg",
    "13_dior_lady.jpg", "14_prada_galleria.jpg", "15_rayban_wayfarer.jpg",
    "16_cartier_love.jpg", "17_cartier_tank.jpg", "18_tommy_crossbody.jpg",
]


def resolve_photo_path(filename):
    """
    Retrouve le fichier photo sur le disque. On tente d'abord le nom exact,
    puis on tolère une extension différente (.jpeg, .png, .webp) ou une
    variation de casse — le renommage automatique au téléchargement est la
    cause d'échec la plus fréquente.
    Retourne le chemin absolu, ou None si rien ne correspond.
    """
    if not filename:
        return None
    exact = os.path.join(PRODUCT_PHOTOS_DIR, filename)
    if os.path.exists(exact):
        return exact
    if not os.path.isdir(PRODUCT_PHOTOS_DIR):
        return None
    prefix = filename.split("_")[0] + "_"      # "01_", "02_", ...
    for f in sorted(os.listdir(PRODUCT_PHOTOS_DIR)):
        if f.lower().startswith(prefix.lower()):
            return os.path.join(PRODUCT_PHOTOS_DIR, f)
    return None


def load_product_photo_b64(filename, brand, model):
    """
    Charge une vraie photo produit depuis static/products/. Force un
    décodage et une reconversion complète en JPEG, quel que soit le format
    réel du fichier (webp/png/avif renommés en .jpg, profils couleur
    exotiques...). Si le fichier est absent ou vraiment illisible, on
    retombe sur le placeholder procédural.
    """
    path = resolve_photo_path(filename)
    if not path:
        return generate_product_photo_b64(brand, model)
    try:
        with Image.open(path) as im:
            im.load()                     # force le décodage complet ici
            img = im.convert("RGB")       # normalise en RGB (gère RGBA, P, CMYK...)
        if max(img.size) > 800:
            img.thumbnail((800, 800), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data = buf.getvalue()
        if len(data) < 1000:              # image suspecte (quasi vide)
            return generate_product_photo_b64(brand, model)
        return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    except Exception as e:
        print(f"[photo] échec {filename}: {e}")   # visible dans les logs
        return generate_product_photo_b64(brand, model)


app = FastAPI(
    title="LUXLOC API",
    description="API de la plateforme de location d'articles de luxe avec vérification IA d'authenticité",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(users.router, prefix="/api/v1/users", tags=["Utilisateurs"])
app.include_router(articles.router, prefix="/api/v1/articles", tags=["Articles"])
app.include_router(verification.router, prefix="/api/v1/verification", tags=["Vérification IA"])
app.include_router(transactions.router, prefix="/api/v1/transactions", tags=["Transactions"])
app.include_router(returns.router, prefix="/api/v1/returns", tags=["Retours"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", tags=["Root"])
def health():
    return {"status": "healthy"}


@app.get("/api/v1/debug/photos", tags=["Root"])
def debug_photos():
    """
    Diagnostic : indique ce que le serveur voit réellement dans
    static/products/. À appeler après chaque déploiement, AVANT de rejouer
    le seed. Si "trouvees" vaut 18, les photos sont bien en place.
    """
    presents, manquantes = [], []
    for f in PHOTO_FILES:
        (presents if resolve_photo_path(f) else manquantes).append(f)
    return {
        "dossier": PRODUCT_PHOTOS_DIR,
        "dossier_existe": os.path.isdir(PRODUCT_PHOTOS_DIR),
        "fichiers_sur_disque": sorted(os.listdir(PRODUCT_PHOTOS_DIR))
        if os.path.isdir(PRODUCT_PHOTOS_DIR) else [],
        "trouvees": len(presents),
        "attendues": len(PHOTO_FILES),
        "manquantes": manquantes,
    }


@app.post("/api/v1/seed", tags=["Root"])
def seed_demo_data():
    db = SessionLocal()
    try:
        def hash_pw(pw):
            return hashlib.sha256(pw.encode()).hexdigest()

        sophie = db.query(User).filter(User.email == "sophie@luxloc.fr").first()
        if sophie:
            return {
                "message": "Donnees de demo deja presentes",
                "users": db.query(User).count(),
                "articles": db.query(Article).count(),
                "note": "Le seed est idempotent : il ne met PAS a jour les photos. "
                        "Pour recharger les photos, utilisez POST /api/v1/reset-and-seed?confirm=yes",
                "login_demo": {"email": "sophie@luxloc.fr", "password": "demo123"},
            }

        sophie = User(email="sophie@luxloc.fr", hashed_password=hash_pw("demo123"),
                      first_name="Sophie", last_name="Martin", role=UserRole.OWNER,
                      is_verified=True, trust_score=0.92)
        emma = User(email="emma@luxloc.fr", hashed_password=hash_pw("demo123"),
                    first_name="Emma", last_name="Dubois", role=UserRole.RENTER,
                    is_verified=True, trust_score=0.88)
        lucas = User(email="lucas@luxloc.fr", hashed_password=hash_pw("demo123"),
                     first_name="Lucas", last_name="Bernard", role=UserRole.BOTH,
                     is_verified=True, trust_score=0.95)
        admin = User(email="admin@luxloc.fr", hashed_password=hash_pw("admin123"),
                     first_name="Admin", last_name="LUXLOC", role=UserRole.ADMIN,
                     is_verified=True, trust_score=1.0)
        db.add_all([sophie, emma, lucas, admin])
        db.flush()

        S, L = sophie.id, lucas.id

        articles_seed = [
            (S, "Speedy Bandoulière 25", "Louis Vuitton", "Speedy 25", ArticleCategory.BAGS, "Toile Monogram, cuir naturel. État impeccable, porté 3 fois.", 85, 1250, "Excellent", True, "LV2024-08742", "14 / 03 / 2024"),
            (S, "Marmont Small Shoulder", "Gucci", "Marmont GG", ArticleCategory.BAGS, "Cuir matelassé noir, quincaillerie dorée.", 65, 980, "Très bon", True, "GG2024-12455", "22 / 02 / 2024"),
            (L, "Submariner Date 41mm", "Rolex", "126610LN", ArticleCategory.WATCHES, "Acier Oystersteel, lunette Cerachrom noire. Révision 2024.", 220, 9500, "Excellent", True, "RX2024-00284", "08 / 01 / 2024"),
            (L, "Kelly 28 Retourné", "Hermès", "Kelly 28 Togo", ArticleCategory.BAGS, "Cuir Togo Gold, quincaillerie Palladium. Livré avec dustbag et boîte.", 195, 8500, "Excellent", True, "HM2024-09812", "17 / 12 / 2023"),
            (S, "Collier Vintage Alhambra", "Van Cleef & Arpels", "Alhambra 10 motifs", ArticleCategory.JEWELRY, "Or jaune 18K, nacre blanche. Certificat inclus.", 110, 4800, "Neuf", True, "VCA2024-00471", "05 / 02 / 2024"),
            (L, "Trench Heritage", "Burberry", "Heritage Classic", ArticleCategory.CLOTHING, "Gabardine de coton miel, taille 38. Ceinture et boutonnage d'origine.", 55, 1800, "Bon", False, "BB2023-44128", "11 / 11 / 2023"),
            (S, "Aviator Classic", "Ray-Ban", "RB3025", ArticleCategory.EYEWEAR, "Monture or, verres verts G-15. Édition originale.", 15, 180, "Excellent", True, "RB2024-09245", "28 / 02 / 2024"),
            (L, "Santos de Cartier", "Cartier", "Santos Medium", ArticleCategory.WATCHES, "Acier et or jaune 18K. Bracelet interchangeable.", 180, 7200, "Excellent", True, "CT2024-00318", "19 / 01 / 2024"),
            (S, "Neverfull MM", "Louis Vuitton", "Neverfull MM Damier", ArticleCategory.BAGS, "Toile Damier Ebene, pochette amovible. Comme neuf.", 70, 1450, "Excellent", True, "LV2024-08901", "02 / 04 / 2024"),
            (L, "Birkin 30 Étoupe", "Hermès", "Birkin 30 Togo", ArticleCategory.BAGS, "Cuir Togo Étoupe, quincaillerie palladiée. Pièce rare.", 380, 12500, "Excellent", True, "HM2023-09650", "20 / 10 / 2023"),
            (S, "Classic Flap Medium", "Chanel", "Classic Flap M", ArticleCategory.BAGS, "Cuir caviar noir, chaîne dorée. État neuf.", 165, 7800, "Neuf", True, "CC2024-44721", "12 / 03 / 2024"),
            (L, "Daytona Cosmograph", "Rolex", "116500LN", ArticleCategory.WATCHES, "Acier, cadran blanc panda, lunette Cerachrom. Avec certificat.", 280, 14500, "Excellent", True, "RX2023-00198", "28 / 09 / 2023"),
            (S, "Lady Dior Medium", "Dior", "Lady Dior M", ArticleCategory.BAGS, "Cuir cannage noir, breloques dorées. Édition 2024.", 145, 5200, "Neuf", True, "DR2024-00892", "08 / 02 / 2024"),
            (L, "Galleria Saffiano", "Prada", "Galleria Saffiano", ArticleCategory.BAGS, "Cuir Saffiano noir, bandoulière amovible.", 85, 2600, "Très bon", True, "PR2024-12076", "29 / 01 / 2024"),
            (S, "Wayfarer Original", "Ray-Ban", "RB2140", ArticleCategory.EYEWEAR, "Écaille de tortue, verres polarisés. Édition limitée.", 18, 220, "Excellent", True, "RB2024-09302", "15 / 03 / 2024"),
            (L, "Love Bracelet", "Cartier", "Love Yellow Gold", ArticleCategory.JEWELRY, "Or jaune 18K, taille 17. Tournevis inclus.", 95, 5800, "Excellent", True, "CT2024-00388", "04 / 02 / 2024"),
            (S, "Tank Française", "Cartier", "Tank Française MM", ArticleCategory.WATCHES, "Acier, cadran argenté. Bracelet acier d'origine.", 130, 4400, "Excellent", True, "CT2024-00425", "21 / 02 / 2024"),
            (L, "Sac Bandoulière Tommy", "Tommy Hilfiger", "Flag Crossbody", ArticleCategory.BAGS, "Toile bleu marine, logo flag tricolore brodé.", 25, 220, "Excellent", True, "TH2024-15028", "17 / 03 / 2024"),
        ]

        articles_objects = []
        for owner, title, brand, model, cat, desc, jour, retail, cond, avail, inv_num, date_str in articles_seed:
            invoice_b64 = generate_invoice_b64(brand, model, cat.value, retail, inv_num, date_str)
            art = Article(
                owner_id=owner, title=title, brand=brand, model=model, category=cat,
                description=desc, price_per_day=jour, retail_price=retail,
                condition=cond, is_available=avail,
                invoice_data=invoice_b64,
            )
            articles_objects.append(art)
        db.add_all(articles_objects)
        db.flush()

        # Photos d'articles — vraie photo depuis static/products/ si elle
        # existe, sinon placeholder procédural.
        photos_reelles = 0
        photo_types = ["overview", "logo", "label", "stitching"]
        for i, art in enumerate(articles_objects):
            photo_file = PHOTO_FILES[i] if i < len(PHOTO_FILES) else None
            if resolve_photo_path(photo_file):
                photos_reelles += 1
            overview_b64 = load_product_photo_b64(photo_file, art.brand, art.model)
            for pt in photo_types:
                db.add(ArticlePhoto(
                    article_id=art.id, photo_type=pt,
                    file_path=f"gs://luxloc-photos/{art.id}/{pt}.jpg",
                    image_data=overview_b64 if pt == "overview" else None,
                ))

        # Indices garantis en "review" (tous avail=True dans le seed ci-dessus)
        # — la file de modération de la console admin ne doit jamais être
        # vide pendant une démonstration.
        REVIEW_INDICES = {2, 9, 15}
        for i, art in enumerate(articles_objects):
            if art.is_available:
                en_examen = (i in REVIEW_INDICES)
                if en_examen:
                    sg = round(random.uniform(0.62, 0.78), 4)
                    status = VerificationStatus.REVIEW
                    art.is_available = False
                else:
                    sg = round(0.86 + random.uniform(0, 0.11), 4)
                    status = VerificationStatus.VERIFIED
                db.add(Verification(
                    article_id=art.id, status=status,
                    score_global=sg,
                    score_logo=round(min(sg + random.uniform(-0.03, 0.05), 1.0), 4),
                    score_stitching=round(min(sg + random.uniform(-0.05, 0.03), 1.0), 4),
                    score_texture=round(min(sg + random.uniform(-0.02, 0.04), 1.0), 4),
                    score_label=round(min(sg + random.uniform(-0.06, 0.02), 1.0), 4),
                    score_rules=round(min(sg + random.uniform(-0.01, 0.05), 1.0), 4),
                    score_invoice=round(0.85 + random.uniform(0, 0.13), 4),
                    pipeline_version="v0.4-ml",
                    processing_time_ms=random.randint(8000, 14000),
                ))

        db.add(Transaction(
            article_id=articles_objects[3].id, renter_id=emma.id,
            status=TransactionStatus.ACTIVE,
            date_start=datetime.utcnow(), date_end=datetime.utcnow() + timedelta(days=3),
            nb_days=3, price_per_day=195, insurance_fee=25, total_amount=610,
            commission=122, guarantee_amount=975,
        ))

        db.commit()
        return {
            "message": "Donnees de demo injectees avec succes !",
            "users": 3,
            "articles": len(articles_objects),
            "factures_generees": len(articles_objects),
            "photos_reelles": photos_reelles,
            "photos_placeholder": len(articles_objects) - photos_reelles,
            "login_demo": {"email": "sophie@luxloc.fr", "password": "demo123"},
        }
    finally:
        db.close()


@app.post("/api/v1/reset-and-seed", tags=["Root"])
def reset_and_seed(confirm: str = ""):
    if confirm != "yes":
        return {
            "message": "Confirmation requise. Ajoutez ?confirm=yes a l'URL.",
            "exemple": "POST /api/v1/reset-and-seed?confirm=yes",
            "attention": "Action DESTRUCTIVE — toutes les donnees seront effacees.",
        }
    db = SessionLocal()
    try:
        deleted = {}
        for model_cls in (ReturnRecord, Transaction, Verification, ArticlePhoto, Article, User):
            n = db.query(model_cls).delete()
            deleted[model_cls.__tablename__] = n
        db.commit()
    finally:
        db.close()
    seed_result = seed_demo_data()
    return {
        "message": "Base reinitialisee et seed execute.",
        "lignes_supprimees": deleted,
        "seed": seed_result,
    }
