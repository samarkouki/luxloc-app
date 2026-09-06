"""
app/admin_dashboard.py

Ajoute une page "Dashboard" avec graphiques dans l'espace /admin,
en plus des vues CRUD déjà fournies par app/admin.py.

Utilise Chart.js + chartjs-plugin-datalabels (via CDN, chargés côté
navigateur) pour :
- Courbe des revenus (commission) par mois
- Barres du nombre de locations par mois
- Camembert de répartition des articles par catégorie (avec %)
- Barres des marques les plus représentées
- Camembert des statuts de transaction (avec %)

Intégration dans app/admin.py :

    from app.admin_dashboard import DashboardView
    ...
    admin.add_view(DashboardView)   # à ajouter EN PREMIER, avant les
                                     # autres add_view, pour qu'il
                                     # apparaisse en tête du menu
"""
import json
from datetime import datetime, timedelta
from string import Template

from sqladmin import BaseView, expose
from sqlalchemy import func
from starlette.requests import Request
from starlette.responses import HTMLResponse

from app.database import SessionLocal
from app.models.models import (
    User, Article, Transaction, TransactionStatus,
    Verification, VerificationStatus,
)


class DashboardView(BaseView):
    name = "Dashboard"
    icon = "fa-solid fa-chart-line"

    @expose("/dashboard", methods=["GET"])
    async def dashboard(self, request: Request):
        db = SessionLocal()
        try:
            since = datetime.utcnow() - timedelta(days=365)

            # ── KPIs ────────────────────────────────────────────────
            total_users = db.query(User).count()
            total_articles = db.query(Article).count()
            total_transactions = db.query(Transaction).count()
            total_commission = db.query(func.sum(Transaction.commission)).filter(
                Transaction.status != TransactionStatus.CANCELLED).scalar() or 0
            verified = db.query(Verification).filter(
                Verification.status == VerificationStatus.VERIFIED).count()
            total_verifs = db.query(Verification).count()
            taux_verif = round(verified / total_verifs * 100, 1) if total_verifs else 0

            # ── Revenu & nombre de locations par mois ────────────────
            txs = (db.query(Transaction.date_start, Transaction.commission, Transaction.status)
                   .filter(Transaction.date_start >= since)
                   .all())
            monthly_revenue, monthly_count = {}, {}
            for date_start, commission, status in txs:
                if status == TransactionStatus.CANCELLED:
                    continue
                key = date_start.strftime("%Y-%m")
                monthly_revenue[key] = monthly_revenue.get(key, 0) + (commission or 0)
                monthly_count[key] = monthly_count.get(key, 0) + 1
            months_sorted = sorted(monthly_revenue.keys())
            revenue_values = [round(monthly_revenue[m], 2) for m in months_sorted]
            count_values = [monthly_count[m] for m in months_sorted]

            # ── Répartition par catégorie ────────────────────────────
            cat_rows = (db.query(Article.category, func.count(Article.id))
                        .group_by(Article.category).all())
            cat_labels = [c.value if hasattr(c, "value") else str(c) for c, _ in cat_rows]
            cat_values = [n for _, n in cat_rows]

            # ── Top marques ───────────────────────────────────────────
            brand_rows = (db.query(Article.brand, func.count(Article.id))
                          .group_by(Article.brand)
                          .order_by(func.count(Article.id).desc())
                          .limit(8).all())
            brand_labels = [b for b, _ in brand_rows]
            brand_values = [n for _, n in brand_rows]

            # ── Statuts des transactions ──────────────────────────────
            status_rows = (db.query(Transaction.status, func.count(Transaction.id))
                          .group_by(Transaction.status).all())
            status_labels = [s.value if hasattr(s, "value") else str(s) for s, _ in status_rows]
            status_values = [n for _, n in status_rows]
        finally:
            db.close()

        html = _TEMPLATE.substitute(
            total_users=total_users,
            total_articles=total_articles,
            total_transactions=total_transactions,
            total_commission=f"{total_commission:,.0f}".replace(",", " "),
            taux_verif=taux_verif,
            revenue_labels=json.dumps(months_sorted),
            revenue_values=json.dumps(revenue_values),
            count_values=json.dumps(count_values),
            cat_labels=json.dumps(cat_labels),
            cat_values=json.dumps(cat_values),
            brand_labels=json.dumps(brand_labels),
            brand_values=json.dumps(brand_values),
            status_labels=json.dumps(status_labels),
            status_values=json.dumps(status_values),
        )
        return HTMLResponse(content=html)


_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>LUXLOC — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-datalabels/2.2.0/chartjs-plugin-datalabels.min.js"></script>
<style>
  body { background:#0f1115; color:#e6e6e6; font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin:0; padding:24px; }
  h1 { font-weight:600; margin-bottom:24px; }
  .kpis { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:32px; }
  .kpi { background:#181b21; border:1px solid #262a33; border-radius:12px; padding:18px 22px; min-width:160px; }
  .kpi .value { font-size:26px; font-weight:700; color:#c9a84c; }
  .kpi .label { font-size:13px; color:#9aa0ab; margin-top:4px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:20px; }
  .card { background:#181b21; border:1px solid #262a33; border-radius:12px; padding:20px; }
  .card h3 { margin-top:0; font-weight:600; font-size:15px; color:#c7c9cf; }
  a.back { color:#c9a84c; text-decoration:none; font-size:14px; }
  @media (max-width:900px){ .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <a class="back" href="/admin">&larr; Retour à l'espace admin</a>
  <h1>Dashboard LUXLOC</h1>

  <div class="kpis">
    <div class="kpi"><div class="value">$total_commission €</div><div class="label">Revenus (commission)</div></div>
    <div class="kpi"><div class="value">$total_transactions</div><div class="label">Locations totales</div></div>
    <div class="kpi"><div class="value">$total_users</div><div class="label">Utilisateurs</div></div>
    <div class="kpi"><div class="value">$total_articles</div><div class="label">Articles au catalogue</div></div>
    <div class="kpi"><div class="value">$taux_verif %</div><div class="label">Taux de vérification auto</div></div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Revenus par mois (commission, €)</h3>
      <canvas id="revenueChart"></canvas>
    </div>
    <div class="card">
      <h3>Nombre de locations par mois</h3>
      <canvas id="countChart"></canvas>
    </div>
    <div class="card">
      <h3>Articles par catégorie</h3>
      <canvas id="catChart"></canvas>
    </div>
    <div class="card">
      <h3>Top marques (nombre d'articles)</h3>
      <canvas id="brandChart"></canvas>
    </div>
    <div class="card">
      <h3>Statuts des transactions</h3>
      <canvas id="statusChart"></canvas>
    </div>
  </div>

<script>
const goldPalette = ["#c9a84c","#8b693c","#4a4a4a","#5a0a0a","#0a3d2a","#2c1654","#1a1a2e","#001f5c"];
Chart.register(ChartDataLabels);

function pctFormatter(value, ctx) {
  const total = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
  return total ? (value / total * 100).toFixed(1) + '%' : '';
}

new Chart(document.getElementById('revenueChart'), {
  type: 'line',
  data: { labels: $revenue_labels, datasets: [{ label: 'Commission (€)', data: $revenue_values,
    borderColor: '#c9a84c', backgroundColor: 'rgba(201,168,76,0.15)', fill: true, tension: 0.3 }] },
  options: {
    plugins: { legend: { display: false }, datalabels: { display: false } },
    scales: { x: { ticks: { color: '#9aa0ab' } }, y: { ticks: { color: '#9aa0ab' } } }
  }
});

new Chart(document.getElementById('countChart'), {
  type: 'bar',
  data: { labels: $revenue_labels, datasets: [{ label: 'Locations', data: $count_values, backgroundColor: '#8b693c' }] },
  options: {
    plugins: { legend: { display: false }, datalabels: { display: false } },
    scales: { x: { ticks: { color: '#9aa0ab' } }, y: { ticks: { color: '#9aa0ab' } } }
  }
});

new Chart(document.getElementById('catChart'), {
  type: 'doughnut',
  data: { labels: $cat_labels, datasets: [{ data: $cat_values, backgroundColor: goldPalette }] },
  options: {
    plugins: {
      legend: { position: 'bottom', labels: { color: '#e6e6e6' } },
      datalabels: { color: '#0f1115', font: { weight: 'bold', size: 12 }, formatter: pctFormatter }
    }
  }
});

new Chart(document.getElementById('brandChart'), {
  type: 'bar',
  data: { labels: $brand_labels, datasets: [{ label: 'Articles', data: $brand_values, backgroundColor: '#c9a84c' }] },
  options: {
    indexAxis: 'y',
    plugins: { legend: { display: false }, datalabels: { display: false } },
    scales: { x: { ticks: { color: '#9aa0ab' } }, y: { ticks: { color: '#9aa0ab' } } }
  }
});

new Chart(document.getElementById('statusChart'), {
  type: 'pie',
  data: { labels: $status_labels, datasets: [{ data: $status_values, backgroundColor: goldPalette }] },
  options: {
    plugins: {
      legend: { position: 'bottom', labels: { color: '#e6e6e6' } },
      datalabels: { color: '#fff', font: { weight: 'bold', size: 12 }, formatter: pctFormatter }
    }
  }
});
</script>
</body>
</html>
""")