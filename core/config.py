"""
Configuracion PRECIARIO_NVDA — fork de PRECIARIO_OIL/core/config.py
Modo principal: DAY TRADING con velas de 1 hora (NVDA target, SOXQ benchmark).
"""
from dataclasses import dataclass
from typing import List
from pathlib import Path
import os
import re

_base_dir = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_base_dir / ".env")
    # Perfil de tu companero: .env define APP_PROFILE y .env.{perfil} sobreescribe.
    _profile = os.environ.get("APP_PROFILE", "").strip()
    if _profile:
        if re.fullmatch(r"[\w-]+", _profile):
            load_dotenv(_base_dir / f".env.{_profile}", override=True)
        else:
            raise ValueError(f"APP_PROFILE invalido (se ignora): {_profile!r}")
except ImportError:
    # python-dotenv no instalado: se continua con las variables del entorno.
    pass

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "core" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_HISTORY_DIR = OUTPUT_DIR / "data_history"
NVDA_HISTORY_DIR = OUTPUT_DIR / "nvda_history"
NEWS_HISTORY_DIR = OUTPUT_DIR / "news_history"
INTRADIA_DIR = OUTPUT_DIR / "intradia"
INTRADIA_BACKUP_DIR = INTRADIA_DIR / "backups"
FEATURES_DIR = OUTPUT_DIR / "features"

DATA_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
NVDA_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
NEWS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
INTRADIA_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_EXCEL = DATA_HISTORY_DIR / "nvda_daily.xlsx"
INTRADIA_FILE = INTRADIA_DIR / "nvda_intradia.xlsx"
INTRADIA_CSV = INTRADIA_DIR / "nvda_intradia.csv"
# Datos expandidos del companero (historico 2020+, velas 1h por simbolo)
INTRADIA_NVDA_CSV = INTRADIA_DIR / "NVDA_intraday.csv"
INTRADIA_SOXQ_CSV = INTRADIA_DIR / "SOXQ_intraday.csv"
# Corte temporal obligatorio: SOXQ inicia 2021-06-11 (descarta pandemia)
DATA_START_DATE = os.environ.get("DATA_START_DATE", "2021-06-11")
# Fuente de datos del pipeline:
#   "db"  -> solo PostgreSQL remoto (Aiven, tabla ohlcv/news); error si no responde
#   "auto"-> DB primero y fallback a CSV/Excel locales si no responde (default)
#   "csv" -> solo CSVs expandidos / Excel IBKR (modo legacy)
DATA_SOURCE = os.environ.get("DATA_SOURCE", "auto").strip().lower()
if DATA_SOURCE not in {"db", "auto", "csv"}:
    DATA_SOURCE = "auto"
FEATURES_CSV = FEATURES_DIR / "nvda_features.csv"
LABELED_CSV = FEATURES_DIR / "nvda_labeled.csv"
SNAPSHOT_CSV = BASE_DIR / "data" / "nvda_snapshot_horario.csv"

NVDA_HISTORY_EXCEL = NVDA_HISTORY_DIR / "nvda_predictions.xlsx"
HISTORY_EXCEL = NVDA_HISTORY_EXCEL

NEWS_HISTORY_JSON = NEWS_HISTORY_DIR / "news_history_nvda.json"
NEWS_HISTORY_EXCEL = NEWS_HISTORY_DIR / "news_history_nvda.xlsx"

WEB_DIR = BASE_DIR / "web" / "REPORT_NVDA"
WEB_DIR.mkdir(parents=True, exist_ok=True)
WEB_HTML_NAME = "index.html"

IBKR_HOST = os.environ.get("IBKR_HOST", "127.0.0.1")
IBKR_RUNTIME = os.environ.get("IBKR_RUNTIME", "tws").lower()
IBKR_DEFAULT_PORT = 4002 if IBKR_RUNTIME in {"gateway", "ibgateway"} else 7497
IBKR_PORT = int(os.environ.get("IBKR_PORT", str(IBKR_DEFAULT_PORT)))
IBKR_CLIENT_ID = int(os.environ.get("IBKR_CLIENT_ID", "11"))
IBKR_INTRADIA_CLIENT_ID = int(os.environ.get("IBKR_INTRADIA_CLIENT_ID", "20"))
IBKR_INTRADIA_TIMEOUT = int(os.environ.get("IBKR_INTRADIA_TIMEOUT", "90"))

ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")

# ============================================================
# INTRADIA (DAY TRADING)
# ============================================================
# Barra de 1 hora: NVDA opera 09:30-16:00 ET = 7 barras/dia RTH
# (8:30-15:00 Lima en DST, 9:30-16:00 Lima en invierno).
INTRADAY_BAR_SIZE = os.environ.get("INTRADAY_BAR_SIZE", "1 hour")
INTRADAY_DURATION_PER_REQ = os.environ.get("INTRADAY_DURATION_PER_REQ", "1 M")
INTRADAY_RECENT_DURATION = os.environ.get("INTRADAY_RECENT_DURATION", "2 D")
INTRADAY_MONTHS_BACK = int(os.environ.get("INTRADAY_MONTHS_BACK", "6"))
INTRADAY_WHAT_TO_SHOW = os.environ.get("INTRADAY_WHAT_TO_SHOW", "TRADES")
INTRADAY_USE_RTH = int(os.environ.get("INTRADAY_USE_RTH", "1"))

DEFAULT_BAR_SIZE = os.environ.get("DEFAULT_BAR_SIZE", "1 hour")

# ============================================================
# DAY TRADING — TARGET / LABELS
# ============================================================
DAYTRADE_HORIZON_BARS = int(os.environ.get("MODEL_HORIZON_BARS", "4"))
MODEL_TARGET_MODE = os.environ.get("MODEL_TARGET_MODE", "simple")  # simple|log
DAYTRADE_TAU_MODE = os.environ.get("MODEL_TAU_MODE", "fixed")
DAYTRADE_TAU_PCT = float(os.environ.get("MODEL_TAU_PCT", "0.8"))
DAYTRADE_TAU_K = float(os.environ.get("MODEL_TAU_K", "0.5"))
DAYTRADE_TAU_MIN = float(os.environ.get("MODEL_TAU_MIN", "0.003"))
DAYTRADE_TAU_MAX = float(os.environ.get("MODEL_TAU_MAX", "0.02"))
MODEL_PURGE_BARS = int(os.environ.get("MODEL_PURGE_BARS", "4"))
MODEL_EMBARGO_BARS = int(os.environ.get("MODEL_EMBARGO_BARS", "7"))

# Features excluidas del modelo por diseno (ver resultados.md 2026-09-20):
#   - pat_*: patrones binarios, ruido medido con 5 anos de datos
#   - cnt_*/has_sentiment_24h: conteo de noticias, captura densidad del dataset
#     (drift 1.6 -> 120 articulos/semana) y rompe el holdout
MODEL_USE_PATTERNS = int(os.environ.get("MODEL_USE_PATTERNS", "0"))
MODEL_USE_NEWS_COUNTS = int(os.environ.get("MODEL_USE_NEWS_COUNTS", "0"))

RSI_PERIOD = int(os.environ.get("MODEL_RSI_PERIOD", "14"))
SMA_PERIODS = [int(x) for x in os.environ.get("MODEL_SMA_PERIODS", "20,50").split(",")]
BB_PERIOD = int(os.environ.get("MODEL_BB_PERIOD", "20"))
ATR_PERIOD = int(os.environ.get("MODEL_ATR_PERIOD", "14"))
BETA_WINDOW = int(os.environ.get("MODEL_BETA_WINDOW", "60"))
SENT_RELEVANCE_MIN = float(os.environ.get("MODEL_SENT_RELEVANCE_MIN", "0.3"))

PRICE_MOVE_THRESHOLD = 0.03
VOLUME_SPIKE_THRESHOLD = 1.8
NEWS_LOOKBACK_HOURS = 24
PATTERN_MIN_BARS = 5
MAX_VISUALIZATIONS = 4

# ============================================================
# ACTIVOS — NASDAQ (mismo close 16:00 ET = 15:00 Lima DST)
# ============================================================
@dataclass
class AssetSpec:
    name: str
    sheet_name: str
    exchange: str
    market_tz: str
    close_time_local: str
    enabled: bool = True
    data_delay_minutes: int = 15


@dataclass
class IntradaySpec:
    """Especificacion de activo para descarga intradia."""
    name: str
    sheet: str
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    primary_exchange: str = "NASDAQ"
    currency: str = "USD"
    enabled: bool = True


ASSETS: List[AssetSpec] = [
    AssetSpec(name="NVDA", sheet_name="NVDA_DAILY", exchange="NASDAQ",
              market_tz="America/New_York", close_time_local="16:00",
              data_delay_minutes=15, enabled=True),
    AssetSpec(name="SOXQ", sheet_name="SOXQ_DAILY", exchange="NASDAQ",
              market_tz="America/New_York", close_time_local="16:00",
              data_delay_minutes=15, enabled=True),
]

INTRADAY_MARKETS: List[IntradaySpec] = [
    IntradaySpec(name="NVDA", sheet="NVDA_1H", symbol="NVDA", enabled=True),
    IntradaySpec(name="SOXQ", sheet="SOXQ_1H", symbol="SOXQ", enabled=True),
]

# Horas de barras intraday RTH (por indice de barra, robusto ante DST)
INTRADAY_BARS_PER_DAY = int(os.environ.get("INTRADAY_BARS_PER_DAY", "7"))
SNAPSHOT_BAR_COUNT = INTRADAY_BARS_PER_DAY

NASDAQ_HOLIDAYS = [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-07-03", "2026-09-07", "2026-11-11",
    "2026-11-26", "2026-12-25",
]
NYMEX_HOLIDAYS = NASDAQ_HOLIDAYS  # alias compatibilidad

# ── Noticias / sentimiento ──
NEWS_WINDOWS_H = [int(x) for x in os.environ.get("NEWS_WINDOWS_H", "6,24,72,168,720").split(",")]
NEWS_HALF_LIFE_H = float(os.environ.get("NEWS_HALF_LIFE_H", "12"))
def _project_path(value: str, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else BASE_DIR / path


NEWS_ARTICLES_FILE = _project_path(
    os.environ.get("NEWS_ARTICLES_PATH", ""),
    "core/output/news_history/news.csv",
)
NEWS_DAILY_FILE = _project_path(
    os.environ.get("NEWS_DAILY_PATH", ""),
    "data/news_nvda_daily.csv",
)

# ============================================================
# BOT / NOTIFICACIONES / MODELO PRODUCTIVO
# ============================================================
# Webhook opcional (Synology Chat u otro): URL completa con token.
NVDA_WEBHOOK_URL = os.environ.get("NVDA_WEBHOOK_URL", "")
NVDA_WEBHOOK_ENABLED = os.environ.get("NVDA_WEBHOOK_ENABLED", "0") == "1"
# Verificacion TLS del webhook: 1 (default) valida el certificado; 0 solo si
# el NAS usa certificado autofirmado (riesgo MITM aceptado explicitamente).
NVDA_WEBHOOK_VERIFY_SSL = os.environ.get("NVDA_WEBHOOK_VERIFY_SSL", "1") == "1"

# Solo emitir alerta push si la probabilidad de BUY/SELL supera este umbral;
# si no, la senal queda como HOLD (menos ruido y menos costes).
SIGNAL_PROBA_THRESHOLD = float(os.environ.get("SIGNAL_PROBA_THRESHOLD", "0.40"))

# Consistencia train/serve: con etiquetado intra-sesion (H barras) el modelo
# solo entrena con las primeras (INTRADAY_BARS_PER_DAY - H) barras de cada dia.
# Si la barra actual esta fuera de ese soporte:
#   "0" -> predice igualmente y avisa (default)
#   "1" -> suprime la senal (label SIN_SENAL)
PREDICT_STRICT_SUPPORT = os.environ.get("PREDICT_STRICT_SUPPORT", "0") == "1"

# Horas/minutos NY en que el bot busca barras recien cerradas
BOT_RUN_MINUTE_WINDOW = int(os.environ.get("BOT_RUN_MINUTE_WINDOW", "5"))
BOT_POLL_SECONDS = int(os.environ.get("BOT_POLL_SECONDS", "60"))
BOT_DRY_RUN = os.environ.get("BOT_DRY_RUN", "1") == "1"

MODELS_DIR = OUTPUT_DIR / "nvda_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas que NUNCA deben entrar al modelo (target, fuga de informacion
# o niveles de precio no estacionarios). Ver core/nvda_trainer.py
NON_FEATURE_COLUMNS = {
    "Datetime", "date_ny", "label", "future_ret", "tau",
    "label_start", "label_end", "label_horizon_bars",
    "Open", "High", "Low", "Close", "Volume", "adj_close",
    "Adj Close",
    "sox_close", "sox_volume", "vwap", "orb_high", "orb_low",
    "sma_20", "sma_50", "bb_mid", "macd", "macd_signal", "macd_hist",
}
