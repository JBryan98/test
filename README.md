# PRECIARIO_NVDA

Sistema predictivo NVDA/SOX para **day trading** con velas de **1 hora**.
Guías: `INSTALL.md` (instalación Ubuntu), `PROYECTO_NVDA.md` (diseño),
`INDICADORES.md` (variables para PPT) y `KPIS.md` (evaluación del modelo y backtest).

## Quick start (Ubuntu)

```bash
bash scripts/setup_ubuntu.sh          # venv + deps + carpetas + .env
nano .env                             # IBKR_HOST, webhook, etc.
source venv/bin/activate

python run_nvda.py --data-only        # 1h NVDA+SOXQ 12 meses exclusivamente IBKR
python run_nvda.py                    # features + labels + snapshot
python run_nvda.py --train            # LogReg + RF + XGB, con ablaciones NVDA/SOX/news
python run_nvda.py --predict --dry-run
python run_nvda.py --bot --dry-run    # bot horario en cada cierre de barra RTH
python run_nvda.py --inspect          # diagnóstico de datos
```

## Pipeline

```
DB remota (ibsync/Docker) ──┐  ohlcv NVDA+SOXQ 1h
DB remota (news) ──────────┴→ features (75 cols · 48 al modelo) → labels BUY/HOLD/SELL
 (fallback: CSVs locales + IBKR directo)         │
                              │
                     ┌────────┴─────────┐
                     ▼                  ▼
              nvda_trainer.py    nvda_predictor.py
               (ablaciones, purged) (umbral → señal)
                     │                  │
                     └────────┬─────────┘
                              ▼
              bot horario + webhook (BI en fase futura)
```

## Estructura

```
core/
  config.py                    # rutas, activos, parámetros 1h/day-trading
  data_providers/db_source.py  # DB remota PostgreSQL (Aiven, SOLO LECTURA): ohlcv + news
  data_providers/ibkr_intraday.py  # descarga 1h chunked + recent desde IBKR (fallback)
  feature_engineering.py       # 75 columnas (48 al modelo): técnicas, sesión/VWAP/ORB, SOX, patrones
  labels.py                    # target por barras + grid tau
  pattern_detector.py          # 57 patrones (heredado de PRECIARIO_OIL)
  news/alpha_vantage.py        # sentimiento point-in-time / lag-1 (sin leakage)
  nvda_trainer.py              # LogReg/RF/XGB + ablaciones + walk-forward purged
  nvda_predictor.py            # inferencia + umbral de señal
  notifications.py             # webhook (Synology u otro)
  nvda_snapshot.py             # backup + snapshot por barra
run_nvda.py                    # orquestador + bot
scripts/setup_ubuntu.sh        # instalador
scripts/dev/test_pipeline_synthetic.py    # pipeline sin TWS
scripts/dev/test_sentiment_alignment.py    # test anti-leakage noticias
KPIS.md                         # KPIs de clasificación y financieros
```

## Estado

- [x] Datos 1h NVDA+SOXQ desde 2021-06-11 (CSV expandido + IBKR)
- [x] Features intradía (técnicas, sesión/VWAP/ORB, SOX, noticias point-in-time)
- [x] Labels day trading (H=4, tau fijo 0.8%, HOLD 49%)
- [x] Modelo LogReg con ablaciones NVDA/SOX/noticias; OOF AUC 0.600, HOLDOUT AUC 0.601
- [x] Predictor con umbral + bot horario + webhook
- [x] Informe HTML de 8 pestañas (`web/REPORT_NVDA/index.html`)
- [ ] Dashboard Power BI (fase futura, fuera de alcance por ahora)
