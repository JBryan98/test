#!/usr/bin/env python
"""
Genera el informe HTML final (8 pestanas) del sistema NVDA day trading.

Contenido:
  1. KPIs financieros (backtest out-of-sample, serie horaria completa)
  2. KPIs generales del modelo (OOF / HOLDOUT)
  3. Datos OHLCV + indicadores 1h (NVDA y SOXQ) + export CSV
  4. Impacto de noticias (heatmap por signo/impacto + sentimiento por anio)
  5. Dashboard descriptivo (estado de mercado + velas avanzadas + noticias)
  6. Dashboard prescriptivo (senal, riesgo, backtest por sesiones, factores)
  7. Auditoria / glosario en 3 secciones
  8. SOXQ y su influencia en NVDA

Backtest (corregido 2026-09-20):
  - Usa TODAS las barras horarias del periodo de test (no solo las etiquetadas;
    las etiquetadas son 3 por sesion, y encadenar sus retornos mezclaba el gap
    nocturno e inflaba los KPIs).
  - Estrategia day trading: una entrada por sesion en la primera senal, salida
    a las H barras o al cierre de sesion (sin overnight).
  - Coste por cambio de posicion configurable; se reporta sensibilidad.

Uso:
    python scripts/report/build_report.py

Salida:
    web/REPORT_NVDA/index.html
    web/REPORT_NVDA/data/*.csv
"""
import html as html_mod
import json
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from core.config import (DATA_START_DATE, DAYTRADE_HORIZON_BARS, DAYTRADE_TAU_PCT,
                         FEATURES_CSV, INTRADAY_BARS_PER_DAY, LABELED_CSV,
                         MODEL_EMBARGO_BARS, MODEL_PURGE_BARS, MODELS_DIR,
                         NEWS_ARTICLES_FILE, PREDICT_STRICT_SUPPORT,
                         SIGNAL_PROBA_THRESHOLD, WEB_DIR, WEB_HTML_NAME)
from core.feature_engineering import (compute_intraday_context,
                                      compute_technical_features,
                                      load_intradia_sheet)
from core.nvda_trainer import (IDX_TO_LABEL, LABEL_TO_IDX, LABELS, _is_sox,
                               build_model, development_test_split, feature_sets,
                               load_dataset, purged_walk_forward)

logger = logging.getLogger("build_report")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BARS_PER_YEAR = 7 * 252            # 1,764 barras RTH
COST_CONSERVATIVE = 0.001          # 0.10% por cambio (supuesto conservador del proyecto)
COST_REALISTA = 0.0002             # 0.02% por cambio (NVDA muy liquido, orden a mercado)
COST_GRID = [0.0001, 0.0002, 0.0005, 0.001]
NY = "America/New_York"
NY_TZ = ZoneInfo(NY)


def ny_str(values) -> pd.Series:
    """Convierte timestamps UTC a hora de Nueva York (formato 'YYYY-MM-DD HH:MM')."""
    s = pd.to_datetime(values, utc=True, errors="coerce")
    return s.dt.tz_convert(NY_TZ).dt.strftime("%Y-%m-%d %H:%M")


# ----------------------------------------------------------------------------
# 1. Modelo: predicciones OOF y holdout
# ----------------------------------------------------------------------------
def _X(df: pd.DataFrame, feats) -> pd.DataFrame:
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _metrics(y_true, y_pred, proba) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "n_eval": int(len(y_true)),
    }
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    for i, lab in enumerate(LABELS):
        out[f"precision_{lab}"] = float(prec[i])
        out[f"recall_{lab}"] = float(rec[i])
        out[f"f1_{lab}"] = float(f1[i])
        out[f"support_{lab}"] = int(sup[i])
    if proba is not None:
        out["roc_auc_ovr"] = float(roc_auc_score(
            y_true, proba, multi_class="ovr", labels=[0, 1, 2]))
    return out


def oof_predictions(name: str, X: pd.DataFrame, y: np.ndarray,
                    purge: int, embargo: int):
    """Walk-forward expanding con purge/embargo. Devuelve pred, proba y AUC por fold."""
    n = len(X)
    pred = np.full(n, -1, dtype=int)
    proba = np.full((n, 3), np.nan)
    fold_auc = []
    for tr, te in purged_walk_forward(n, n_splits=5, purge=purge,
                                      embargo=embargo, min_train_fraction=0.5):
        model = build_model(name)
        model.fit(X.iloc[tr], y[tr])
        pred[te] = model.predict(X.iloc[te])
        proba[te] = model.predict_proba(X.iloc[te])
        try:
            fold_auc.append(roc_auc_score(y[te], proba[te], multi_class="ovr", labels=[0, 1, 2]))
        except ValueError:
            pass
    return pred, proba, fold_auc


def evaluate_sets(df: pd.DataFrame) -> dict:
    """Evalua OOF + holdout para los ablations A/B/D con la config vigente."""
    dev, test, test_start = development_test_split(df, test_fraction=0.2)
    sets = feature_sets(df)
    out = {}
    for tag in ["A_nvda_only", "B_nvda_sox", "D_nvda_sox_news"]:
        feats = sets[tag]
        y_dev = dev["label"].map(LABEL_TO_IDX).to_numpy()
        y_test = test["label"].map(LABEL_TO_IDX).to_numpy()
        Xd, Xt = _X(dev, feats), _X(test, feats)
        pred, proba, folds = oof_predictions("logreg", Xd, y_dev,
                                             MODEL_PURGE_BARS, MODEL_EMBARGO_BARS)
        mask = pred >= 0
        oof = _metrics(y_dev[mask], pred[mask], proba[mask])
        oof["auc_folds_mean"] = float(np.mean(folds)) if folds else np.nan
        oof["auc_folds_std"] = float(np.std(folds)) if folds else np.nan
        model = build_model("logreg")
        model.fit(Xd, y_dev)
        pred_t = model.predict(Xt)
        proba_t = model.predict_proba(Xt)
        hold = _metrics(y_test, pred_t, proba_t)
        out[tag] = {
            "features": feats,
            "n_features": len(feats),
            "oof": oof,
            "holdout": hold,
            "test_start": str(test_start),
            "n_dev": int(len(dev)),
            "n_test": int(len(test)),
            "dev": dev,
            "test": test,
            "holdout_pred": pred_t,
            "holdout_proba": proba_t,
            "oof_pred": pred,
            "oof_proba": proba,
        }
        logger.info(f"{tag}: {len(feats)} feat | OOF AUC {oof.get('roc_auc_ovr'):.4f} "
                    f"F1 {oof['f1_macro']:.4f} | HOLD AUC {hold.get('roc_auc_ovr'):.4f} "
                    f"F1 {hold['f1_macro']:.4f}")
    return out


# ----------------------------------------------------------------------------
# 2. Backtest financiero (holdout, serie horaria completa)
# ----------------------------------------------------------------------------
def _equity(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0


def _sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    return float(returns.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0


def _tstat(returns: np.ndarray) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(len(returns))) if std > 0 else 0.0


def _trade_returns(pos: np.ndarray, ret: np.ndarray) -> list:
    trades, cur, in_pos = [], 1.0, False
    for i in range(len(pos)):
        if i > 0 and pos[i] != pos[i - 1]:
            if in_pos:
                trades.append(cur - 1.0)
            cur, in_pos = 1.0, pos[i] != 0
        elif not in_pos and pos[i] != 0:
            in_pos, cur = True, 1.0
        if pos[i] != 0:
            cur *= (1.0 + ret[i])
    if in_pos:
        trades.append(cur - 1.0)
    return trades


def make_signals(proba: np.ndarray, threshold: float) -> np.ndarray:
    """BUY/SELL solo si superan el umbral de senal; si no HOLD. Devuelve +1/0/-1."""
    sig = np.zeros(len(proba), dtype=int)
    p_buy, p_sell = proba[:, 0], proba[:, 2]
    buy = (p_buy >= threshold) & (p_buy >= proba[:, 1]) & (p_buy >= p_sell)
    sell = (p_sell >= threshold) & (p_sell >= proba[:, 1]) & (p_sell > p_buy)
    sig[buy] = 1
    sig[sell] = -1
    return sig


def build_test_bars(feats_df: pd.DataFrame, test_sessions) -> pd.DataFrame:
    """Todas las barras horarias (7/dia) de las sesiones del holdout."""
    mask = feats_df["date_ny"].astype(str).isin({str(s) for s in test_sessions})
    bars = feats_df[mask].reset_index(drop=True).copy()
    d = bars["date_ny"].to_numpy()
    bars["last_of_session"] = np.append(d[1:] != d[:-1], True)
    bars["first_of_session"] = np.concatenate([[True], d[1:] != d[:-1]])
    return bars


def positions_one_entry(sig: np.ndarray, last_of_session: np.ndarray,
                        first_of_session: np.ndarray, hold: int = 4,
                        allow_short: bool = False) -> np.ndarray:
    """Una entrada por sesion (primera senal); salida a las `hold` barras o al cierre."""
    n = len(sig)
    pos = np.zeros(n, dtype=int)
    cur, entry, done = 0, -1, False
    for i in range(n):
        if first_of_session[i]:
            done = False
        if cur != 0 and (last_of_session[i] or (i - entry) >= hold):
            cur = 0
        if cur == 0 and not done:
            if sig[i] == 1:
                cur, entry, done = 1, i, True
            elif sig[i] == -1 and allow_short:
                cur, entry, done = -1, i, True
        pos[i] = cur
    return pos


def backtest_bars(bars: pd.DataFrame, pos: np.ndarray, cost: float) -> dict:
    """Retornos horarios con ejecucion en la barra siguiente y coste por cambio."""
    close = bars["adj_close"].astype(float).to_numpy()
    ret_bar = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    pos_exec = np.concatenate([[0], pos[:-1]])
    changes = np.abs(np.diff(np.concatenate([[0.0], pos_exec])))
    strat = pos_exec * ret_bar - changes * cost
    eq = _equity(strat)
    bh = _equity(ret_bar)
    trades = _trade_returns(pos_exec, strat)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    years = max(len(strat), 1) / BARS_PER_YEAR
    total = float(eq[-1] - 1.0)
    return {
        "returns": strat,
        "positions": pos_exec,
        "equity": eq,
        "buy_hold_equity": bh,
        "buy_hold_return": float(bh[-1] - 1.0),
        "sharpe": _sharpe(strat),
        "sortino": _sortino(strat),
        "tstat": _tstat(strat),
        "max_dd": _max_drawdown(eq),
        "calmar": float(total / abs(_max_drawdown(eq))) if _max_drawdown(eq) < 0 else np.nan,
        "total_return": total,
        "annual_return": float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan,
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else np.nan,
        "profit_factor": float(sum(wins) / abs(sum(losses))) if losses and sum(losses) else np.nan,
        "avg_trade": float(np.mean(trades)) if trades else np.nan,
        "total_costs": float(changes.sum() * cost),
        "active_bars": int((pos_exec != 0).sum()),
        "trades": trades,
    }


# ----------------------------------------------------------------------------
# 3. Noticias
# ----------------------------------------------------------------------------
def sentiment_from_score(score: float) -> str:
    if score >= 0.35:
        return "Bullish"
    if score >= 0.15:
        return "Somewhat-Bullish"
    if score > -0.15:
        return "Neutral"
    if score > -0.35:
        return "Somewhat-Bearish"
    return "Bearish"


def load_news() -> pd.DataFrame:
    """Noticias para el informe: DB remota primero (DATA_SOURCE), CSV como fallback."""
    from core.config import DATA_SOURCE

    news = None
    if DATA_SOURCE in ("db", "auto"):
        try:
            from core.data_providers.db_source import load_news_articles
            arts = load_news_articles()
            if arts is not None and not arts.empty:
                news = arts.rename(columns={"score": "sentiment_score",
                                            "label": "sentiment_label"})
            elif DATA_SOURCE == "db":
                raise RuntimeError("DATA_SOURCE=db y la DB no devolvio articulos")
        except Exception as e:
            if DATA_SOURCE == "db":
                raise
            logger.warning(f"Noticias: DB no disponible ({e}) - fallback a CSV")

    if news is None:
        news = pd.read_csv(NEWS_ARTICLES_FILE, low_memory=False)
        news["published_utc"] = pd.to_datetime(news["time_published"], utc=True, errors="coerce")

    news = news.dropna(subset=["published_utc"])
    news["ticker"] = news["ticker"].astype(str).str.upper()
    news["relevance_score"] = pd.to_numeric(news["relevance_score"], errors="coerce").fillna(0.0)
    news["sentiment_score"] = pd.to_numeric(news["sentiment_score"], errors="coerce").fillna(0.0)
    news = news[(news["ticker"] == "NVDA") & (news["relevance_score"] >= 0.3)]
    news = news.sort_values("published_utc").drop_duplicates(subset=["title"], keep="first")
    news["polarity"] = np.where(news["sentiment_score"] > 0.15, "Positiva",
                                np.where(news["sentiment_score"] < -0.15, "Negativa", "Neutral"))
    news["impact_score"] = news["relevance_score"] * news["sentiment_score"].abs()
    if len(news["impact_score"].dropna()) > 10:
        q_hi = news["impact_score"].quantile(0.667)
        q_lo = news["impact_score"].quantile(0.333)
        news["impact_level"] = np.where(news["impact_score"] >= q_hi, "Alto",
                                        np.where(news["impact_score"] >= q_lo, "Medio", "Bajo"))
    else:
        news["impact_level"] = "Bajo"
    news["sentiment_label"] = news["sentiment_label"].replace("", np.nan)
    news["sentiment_label_es"] = news["sentiment_label"].fillna(
        news["sentiment_score"].map(sentiment_from_score))
    return news


# ----------------------------------------------------------------------------
# 4. HTML helpers
# ----------------------------------------------------------------------------
def esc(value) -> str:
    return html_mod.escape(str(value))


def table(df: pd.DataFrame, cols=None, rename=None, formatters=None,
          classes: str = "data-table", limit: int = None) -> str:
    view = df[cols] if cols else df
    if limit:
        view = view.tail(limit)
    rename = rename or {}
    view = view.rename(columns=rename)
    return view.to_html(index=False, classes=classes, border=0, escape=True,
                        formatters=formatters, na_rep="—")


def fmt(x, dec=2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:,.{dec}f}"


def pct(x, dec=1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x * 100:,.{dec}f}%"


BADGES = {
    "<td>Alto</td>": '<td><span class="badge impact-alto">Alto</span></td>',
    "<td>Medio</td>": '<td><span class="badge impact-medio">Medio</span></td>',
    "<td>Bajo</td>": '<td><span class="badge impact-bajo">Bajo</span></td>',
    "<td>Bullish</td>": '<td><span class="badge sent-bull">Bullish</span></td>',
    "<td>Somewhat-Bullish</td>": '<td><span class="badge sent-sbull">Somewhat-Bullish</span></td>',
    "<td>Neutral</td>": '<td><span class="badge sent-neutral">Neutral</span></td>',
    "<td>Somewhat-Bearish</td>": '<td><span class="badge sent-sbear">Somewhat-Bearish</span></td>',
    "<td>Bearish</td>": '<td><span class="badge sent-bear">Bearish</span></td>',
}


def badges(html_table: str) -> str:
    for token, replacement in BADGES.items():
        html_table = html_table.replace(token, replacement)
    return html_table


def confusion_html(cm, labels=LABELS) -> str:
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    head = "".join(f"<th>{lab}</th>" for lab in labels)
    rows = ""
    for i, lab in enumerate(labels):
        cells = ""
        for j in range(len(labels)):
            v = norm[i, j]
            shade = f"background: rgba(118,185,0,{0.08 + 0.55 * v:.3f});"
            cells += f'<td style="{shade}">{v * 100:.0f}%<span class="cm-n">({int(cm[i, j])})</span></td>'
        rows += f"<tr><th>{lab}</th>{cells}</tr>"
    return (f'<table class="cm-table"><thead><tr><th>Real \\ Pred</th>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table>")


def metric_row(name, oof, hold, key, dec=4, is_pct=False):
    f = pct if is_pct else (lambda v: fmt(v, dec))
    return (f"<tr><td>{esc(name)}</td><td class='mono'>{f(oof.get(key))}</td>"
            f"<td class='mono'>{f(hold.get(key))}</td></tr>")


# ----------------------------------------------------------------------------
# 5. Datos para el dashboard y el prescriptivo
# ----------------------------------------------------------------------------
def market_state(row: pd.Series) -> tuple:
    score, reasons = 0, []
    close = row["adj_close"]
    if close > row["sma_20"]:
        score += 1; reasons.append("precio sobre SMA20")
    else:
        score -= 1; reasons.append("precio bajo SMA20")
    if close > row["sma_50"]:
        score += 1; reasons.append("precio sobre SMA50")
    else:
        score -= 1; reasons.append("precio bajo SMA50")
    if row["sma_20"] > row["sma_50"]:
        score += 1; reasons.append("SMA20 > SMA50")
    else:
        score -= 1; reasons.append("SMA20 < SMA50")
    if row["macd_hist"] > 0:
        score += 1; reasons.append("MACD positivo")
    else:
        score -= 1; reasons.append("MACD negativo")
    if row["rsi_14"] > 50:
        score += 1; reasons.append("RSI > 50")
    else:
        score -= 1; reasons.append("RSI < 50")
    if score >= 3:
        state, tone = "BULLISH", "tone-buy"
    elif score <= -3:
        state, tone = "BEARISH", "tone-sell"
    else:
        state, tone = "NEUTRAL / LATERAL", "tone-hold"
    return state, tone, score, reasons


def risk_level(atr_pct_series: pd.Series, current: float) -> tuple:
    hist = atr_pct_series.tail(252)
    pctile = float((hist <= current).mean())
    if pctile < 0.33:
        return "BAJO", pctile, "tone-buy"
    if pctile < 0.66:
        return "MODERADO", pctile, "tone-hold"
    return "ALTO", pctile, "tone-sell"


def recommendation_factors(row: pd.Series, news: pd.DataFrame, last_dt) -> list:
    close = row["adj_close"]
    trend = "Alcista" if (close > row["sma_20"] > row["sma_50"]) else (
        "Bajista" if (close < row["sma_20"] < row["sma_50"]) else "Lateral")
    tech = []
    tech.append("RSI sobrecompra" if row["rsi_14"] > 70 else
                "RSI sobreventa" if row["rsi_14"] < 30 else "RSI neutral")
    tech.append("MACD alcista" if row["macd_hist"] > 0 else "MACD bajista")
    tech.append("precio sobre banda media" if row["bb_pctb"] > 0.5 else "precio bajo banda media")
    recent = news[news["published_utc"] <= last_dt]
    w24 = recent[recent["published_utc"] >= last_dt - pd.Timedelta(hours=24)]
    w7d = recent[recent["published_utc"] >= last_dt - pd.Timedelta(days=7)]
    sent24 = w24["sentiment_score"].mean() if len(w24) else 0.0
    vol_trend = "en aumento" if row["vol_zscore"] > 0.5 else (
        "en caida" if row["vol_zscore"] < -0.5 else "estable")
    return [
        ("Tendencia", trend, "tone-buy" if trend == "Alcista" else
         "tone-sell" if trend == "Bajista" else "tone-hold"),
        ("Indicadores tecnicos", " · ".join(tech),
         "tone-buy" if row["macd_hist"] > 0 and row["rsi_14"] >= 50 else
         "tone-sell" if row["macd_hist"] < 0 and row["rsi_14"] < 50 else "tone-hold"),
        ("Sentimiento de noticias",
         f"{sentiment_from_score(sent24)} (24h: {sent24:+.3f} · 7d: "
         f"{(w7d['sentiment_score'].mean() if len(w7d) else 0):+.3f})",
         "tone-buy" if sent24 > 0.15 else "tone-sell" if sent24 < -0.15 else "tone-hold"),
        ("Volumen", f"{vol_trend} (z = {row['vol_zscore']:+.2f})",
         "tone-buy" if row["vol_zscore"] > 0.5 else
         "tone-sell" if row["vol_zscore"] < -0.5 else "tone-hold"),
    ]


def sox_coefficients(dev: pd.DataFrame, feats: list) -> pd.DataFrame:
    model = build_model("logreg")
    y = dev["label"].map(LABEL_TO_IDX).to_numpy()
    model.fit(_X(dev, feats), y)
    coef = model.named_steps["clf"].coef_
    rows = []
    for i, f in enumerate(feats):
        if _is_sox(f):
            rows.append({"variable": f, "coef_BUY": coef[0, i],
                         "coef_HOLD": coef[1, i], "coef_SELL": coef[2, i]})
    out = pd.DataFrame(rows)
    return out.reindex(out[["coef_BUY", "coef_SELL"]].abs().max(axis=1)
                       .sort_values(ascending=False).index)


# ----------------------------------------------------------------------------
# 6. Render
# ----------------------------------------------------------------------------
def render(data: dict, assets: dict) -> str:
    css = (Path(__file__).parent / "report.css").read_text(encoding="utf-8")
    js = (Path(__file__).parent / "report.js").read_text(encoding="utf-8")
    body = (Path(__file__).parent / "report_body.html").read_text(encoding="utf-8")
    html_out = body.replace("/*__CSS__*/", css).replace("/*__JS__*/", js)
    html_out = html_out.replace("__REPORT_DATA__", json.dumps(assets, ensure_ascii=False))
    for key, value in data.items():
        html_out = html_out.replace(f"__{key}__", value)
    return html_out


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------
def main() -> int:
    logger.info("Cargando dataset etiquetado...")
    df = load_dataset()
    feats_df = pd.read_csv(FEATURES_CSV)
    feats_df["Datetime"] = pd.to_datetime(feats_df["Datetime"], utc=True)
    feats_df = feats_df.sort_values("Datetime").reset_index(drop=True)
    # Si la ultima barra aun no ha cerrado (datos descargados a mitad de sesion),
    # se excluye para la senal y los indicadores en vivo.
    now_utc = pd.Timestamp.now(tz="UTC")
    if feats_df["Datetime"].iloc[-1] + pd.Timedelta(hours=1) > now_utc:
        logger.info(f"Barra {feats_df['Datetime'].iloc[-1]} en curso: se excluye del estado en vivo")
        feats_df = feats_df.iloc[:-1].reset_index(drop=True)
    feats_df["date_ny"] = (feats_df["Datetime"].dt.tz_convert(NY_TZ).dt.date)
    last_row = feats_df.iloc[-1]
    last_dt = last_row["Datetime"]

    logger.info("Evaluando feature sets A / B / D (OOF + holdout)...")
    evals = evaluate_sets(df)
    dev, test = evals["D_nvda_sox_news"]["dev"], evals["D_nvda_sox_news"]["test"]
    set_d = evals["D_nvda_sox_news"]
    feats_d = set_d["features"]
    proba_test = set_d["holdout_proba"]
    labels_test = test["label"].to_numpy()

    # ---------------- backtest corregido sobre barras horarias completas ----------------
    logger.info("Backtest day-trading (barras horarias completas, cierre diario)...")
    sessions = sorted(test["date_ny"].astype(str).unique())
    bars_test = build_test_bars(feats_df, sessions)
    sig_labeled = make_signals(proba_test, SIGNAL_PROBA_THRESHOLD)
    sig_map = dict(zip(pd.to_datetime(test["Datetime"], utc=True), sig_labeled))
    bars_test["signal"] = bars_test["Datetime"].map(sig_map).fillna(0).astype(int)
    sig_bars = bars_test["signal"].to_numpy()
    last_s = bars_test["last_of_session"].to_numpy()
    first_s = bars_test["first_of_session"].to_numpy()

    pos_long = positions_one_entry(sig_bars, last_s, first_s,
                                   hold=DAYTRADE_HORIZON_BARS, allow_short=False)
    pos_ls = positions_one_entry(sig_bars, last_s, first_s,
                                 hold=DAYTRADE_HORIZON_BARS, allow_short=True)
    sig_argmax = np.where(proba_test.argmax(axis=1) == 0, 1,
                          np.where(proba_test.argmax(axis=1) == 2, -1, 0))
    map_argmax = dict(zip(pd.to_datetime(test["Datetime"], utc=True), sig_argmax))
    bars_test["signal_argmax"] = bars_test["Datetime"].map(map_argmax).fillna(0).astype(int)
    pos_raw = positions_one_entry(bars_test["signal_argmax"].to_numpy(), last_s, first_s,
                                  hold=DAYTRADE_HORIZON_BARS, allow_short=False)

    bt_long = backtest_bars(bars_test, pos_long, COST_CONSERVATIVE)
    bt_ls = backtest_bars(bars_test, pos_ls, COST_CONSERVATIVE)
    bt_raw = backtest_bars(bars_test, pos_raw, COST_CONSERVATIVE)
    bt_bh = backtest_bars(bars_test, np.ones(len(bars_test), dtype=int), 0.0)
    bt_long_real = backtest_bars(bars_test, pos_long, COST_REALISTA)
    bt_ls_real = backtest_bars(bars_test, pos_ls, COST_REALISTA)
    ic = float(spearmanr(proba_test[:, 0] - proba_test[:, 2],
                         test["future_ret"].to_numpy()).statistic)

    cost_sens_rows = ""
    for cost in COST_GRID:
        bl = backtest_bars(bars_test, pos_long, cost)
        bs = backtest_bars(bars_test, pos_ls, cost)
        cost_sens_rows += (
            f"<tr><td class='mono'>{cost * 100:.2f}%</td>"
            f"<td class='mono'>{bl['sharpe']:.2f}</td><td class='mono'>{pct(bl['total_return'])}</td>"
            f"<td class='mono'>{pct(bl['total_costs'])}</td>"
            f"<td class='mono'>{bs['sharpe']:.2f}</td><td class='mono'>{pct(bs['total_return'])}</td>"
            f"<td class='mono'>{pct(bs['total_costs'])}</td></tr>")

    logger.info("Procesando noticias...")
    news = load_news()
    news_hist = news[news["published_utc"] <= last_dt]
    news_30d = news_hist[news_hist["published_utc"] >= last_dt - pd.Timedelta(days=30)]
    sent_avg_24h = news_hist[news_hist["published_utc"] >= last_dt - pd.Timedelta(hours=24)][
        "sentiment_score"].mean()
    sent_avg_7d = news_hist[news_hist["published_utc"] >= last_dt - pd.Timedelta(days=7)][
        "sentiment_score"].mean()

    state, state_tone, state_score, state_reasons = market_state(last_row)
    risk, risk_pctile, risk_tone = risk_level(feats_df["atr_pct"], last_row["atr_pct"])

    import joblib
    prod_model = joblib.load(MODELS_DIR / "modelo_nvda_logreg.pkl")
    prod_feats = json.loads((MODELS_DIR / "feature_list_logreg.json").read_text(encoding="utf-8"))
    p_live = prod_model.predict_proba(_X(feats_df.tail(1), prod_feats))[0]
    live_label = IDX_TO_LABEL[int(np.argmax(p_live))]
    live_signal_model = "HOLD"
    if live_label == "BUY" and p_live[0] >= SIGNAL_PROBA_THRESHOLD:
        live_signal_model = "BUY"
    elif live_label == "SELL" and p_live[2] >= SIGNAL_PROBA_THRESHOLD:
        live_signal_model = "SELL"

    # Politica de soporte (consistente con core/nvda_predictor.py): el modelo
    # solo entreno con las primeras (INTRADAY_BARS_PER_DAY - H) barras de cada
    # sesion. Fuera de ese soporte la senal no esta validada y, con
    # PREDICT_STRICT_SUPPORT, no se emite (SIN_SENAL).
    max_supported = INTRADAY_BARS_PER_DAY - DAYTRADE_HORIZON_BARS
    bar_of_day = int(last_row["bar_of_day"]) if "bar_of_day" in feats_df.columns else None
    fuera_de_soporte = bar_of_day is not None and bar_of_day > max_supported
    live_signal = "SIN_SENAL" if (fuera_de_soporte and PREDICT_STRICT_SUPPORT) else live_signal_model

    # ---------------- 7 sesiones de backtest ----------------
    last_sessions = sessions[-7:]
    mask7 = bars_test["date_ny"].astype(str).isin(last_sessions)
    eq7 = bt_long["equity"][mask7.to_numpy()]
    bt7 = {
        "dates": ny_str(bars_test["Datetime"][mask7]).tolist(),
        "close": bars_test["adj_close"][mask7].round(2).tolist(),
        "signal": [IDX_TO_LABEL[i] for i in
                   bars_test["signal_argmax"][mask7].map({1: 0, -1: 2}).fillna(1).astype(int)],
        "position": bt_long["positions"][mask7.to_numpy()].tolist(),
        "position_ls": bt_ls["positions"][mask7.to_numpy()].tolist(),
        "equity": (eq7 / eq7[0]).round(5).tolist() if len(eq7) else [],
    }
    bt7["signal"] = ["BUY" if s == 1 else "SELL" if s == -1 else "HOLD"
                     for s in bars_test["signal"][mask7].tolist()]
    bt7["direction"] = ["LONG" if p > 0 else "SHORT" if p < 0 else "FLAT"
                        for p in bt7["position_ls"]]

    # ---------------- tablas ----------------
    kpi_model_rows = "".join([
        metric_row("F1 macro", set_d["oof"], set_d["holdout"], "f1_macro"),
        metric_row("ROC-AUC OVR", set_d["oof"], set_d["holdout"], "roc_auc_ovr"),
        metric_row("Accuracy", set_d["oof"], set_d["holdout"], "accuracy"),
        metric_row("Balanced accuracy", set_d["oof"], set_d["holdout"], "balanced_accuracy"),
        metric_row("Precision BUY", set_d["oof"], set_d["holdout"], "precision_BUY"),
        metric_row("Recall BUY", set_d["oof"], set_d["holdout"], "recall_BUY"),
        metric_row("Precision SELL", set_d["oof"], set_d["holdout"], "precision_SELL"),
        metric_row("Recall SELL", set_d["oof"], set_d["holdout"], "recall_SELL"),
        metric_row("F1 BUY", set_d["oof"], set_d["holdout"], "f1_BUY"),
        metric_row("F1 HOLD", set_d["oof"], set_d["holdout"], "f1_HOLD"),
        metric_row("F1 SELL", set_d["oof"], set_d["holdout"], "f1_SELL"),
    ])
    ablation_rows = "".join(
        f"<tr><td>{esc(tag)}</td><td class='mono'>{m['n_features']}</td>"
        f"<td class='mono'>{m['oof']['f1_macro']:.4f}</td>"
        f"<td class='mono'>{m['oof'].get('roc_auc_ovr', float('nan')):.4f}</td>"
        f"<td class='mono'>{m['holdout']['f1_macro']:.4f}</td>"
        f"<td class='mono'>{m['holdout'].get('roc_auc_ovr', float('nan')):.4f}</td></tr>"
        for tag, m in evals.items())

    key_ind = [
        ("RSI (14)", f"{last_row['rsi_14']:.2f}"),
        ("MACD", f"{last_row['macd']:+.3f}"),
        ("MACD señal", f"{last_row['macd_signal']:+.3f}"),
        ("MACD histograma", f"{last_row['macd_hist']:+.3f}"),
        ("Bollinger %B", f"{last_row['bb_pctb']:.2f}"),
        ("Bollinger ancho", pct(last_row["bb_width"])),
        ("ATR (14)", f"{last_row['atr_14']:.2f}"),
        ("ATR %", pct(last_row["atr_pct"])),
        ("Volumen z-score", f"{last_row['vol_zscore']:+.2f}"),
        ("VWAP distancia", pct(last_row["vwap_dist"])),
        ("Dist. ORB high", pct(last_row["dist_orb_high"])),
        ("Dist. ORB low", pct(last_row["dist_orb_low"])),
        ("Beta 60b", f"{last_row['beta_60b']:.2f}"),
        ("Correlación 60b", f"{last_row['corr_60b']:.2f}"),
        ("Alpha 1 barra", pct(last_row["alpha_1b"])),
        ("SMA 20 / 50", f"{last_row['sma_20']:.2f} / {last_row['sma_50']:.2f}"),
    ]
    key_ind_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='mono'>{v}</td></tr>" for k, v in key_ind)

    factors = recommendation_factors(last_row, news, last_dt)
    factor_rows = "".join(
        f'<div class="factor"><div class="factor-name">{esc(k)}</div>'
        f'<div class="factor-value {tone}">{esc(v)}</div></div>'
        for k, v, tone in factors)

    sig_class = live_signal_model.lower()
    success_p = {"BUY": float(p_live[0]), "HOLD": float(p_live[1]), "SELL": float(p_live[2])}[live_signal_model]
    success_hist = {
        "BUY": set_d["holdout"]["precision_BUY"],
        "HOLD": set_d["holdout"]["precision_HOLD"],
        "SELL": set_d["holdout"]["precision_SELL"],
    }[live_signal_model]

    # ---------------- noticias ----------------
    pos_news = news_hist[news_hist["polarity"] == "Positiva"].sort_values(
        "impact_score", ascending=False).head(120).copy()
    neg_news = news_hist[news_hist["polarity"] == "Negativa"].sort_values(
        "impact_score", ascending=False).head(120).copy()
    recent_news = news_30d.sort_values("published_utc", ascending=False).head(150).copy()
    for frame in (pos_news, neg_news, recent_news):
        frame["published_utc"] = ny_str(frame["published_utc"])
    news_cols = ["published_utc", "source", "title", "sentiment_label_es",
                 "sentiment_score", "relevance_score", "impact_score", "impact_level"]
    news_rename = {"published_utc": "Fecha/hora (ET)", "source": "Fuente", "title": "Título",
                   "sentiment_label_es": "Sentimiento", "sentiment_score": "Score",
                   "relevance_score": "Relevancia", "impact_score": "Impacto",
                   "impact_level": "Nivel"}
    pos_table = badges(table(pos_news, news_cols, news_rename, limit=None))
    neg_table = badges(table(neg_news, news_cols, news_rename, limit=None))
    recent_news_table = badges(table(recent_news, news_cols, news_rename))

    cut_utc = pd.Timestamp(DATA_START_DATE, tz="UTC")
    news_cut = news[news["published_utc"] >= cut_utc]
    news_year = news_cut.assign(year_ny=news_cut["published_utc"].dt.tz_convert(NY_TZ).dt.year)
    sent_year = news_year.groupby("year_ny").agg(
        avg_sent=("sentiment_score", "mean"),
        n=("sentiment_score", "size"),
    ).reset_index()

    levels = ["Alto", "Medio", "Bajo"]
    pols = ["Positiva", "Negativa", "Neutral"]
    news_matrix = {
        "rows": pols,
        "cols": levels,
        "vals": [[int(((news["polarity"] == p) & (news["impact_level"] == l)).sum())
                  for l in levels] for p in pols],
    }

    bt7_table = table(pd.DataFrame({
        "Fecha/hora (ET)": bt7["dates"],
        "Precio": bt7["close"],
        "Señal": bt7["signal"],
        "Dirección (L/S)": bt7["direction"],
        "Equity long-only (base 1)": bt7["equity"],
    }), formatters={"Equity long-only (base 1)": lambda v: f"{v:.4f}"})

    # ---------------- OHLCV + indicadores ----------------
    price_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    ind_cols = ["ret_1b", "log_ret_1b", "ret_3b", "ret_1d", "sma_20", "sma_50",
                "dist_sma_20", "dist_sma_50", "macd", "macd_signal", "macd_hist",
                "rsi_14", "bb_pctb", "bb_width", "atr_14", "atr_pct", "vol_zscore",
                "volume_surge", "hl_range_pct", "close_pos", "vwap", "vwap_dist",
                "orb_high", "orb_low", "dist_orb_high", "dist_orb_low",
                "orb_break_up", "orb_break_down", "cum_day_ret", "gap_open_pct",
                "sox_ret_1b", "sox_ret_1d", "sox_ratio", "sox_ratio_z", "beta_60b",
                "corr_60b", "alpha_1b", "alpha_5b", "rel_strength_1d"]
    shared = [c for c in price_cols if c in feats_df.columns]
    show_nvda = [c for c in shared + ["bar_of_day", "hour_ny", "rsi_14", "macd_hist",
                                      "bb_pctb", "atr_pct", "vol_zscore", "vwap_dist",
                                      "dist_orb_high", "dist_orb_low", "sox_ratio_z",
                                      "beta_60b", "corr_60b", "alpha_1b"] if c in feats_df.columns]
    nvda_view = feats_df[show_nvda].tail(250).copy()
    nvda_view["Datetime"] = ny_str(nvda_view["Datetime"])
    nvda_table = table(nvda_view, rename={
        "Datetime": "Fecha/hora (ET)", "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Volume": "Volumen"})

    soxq = load_intradia_sheet("SOXQ_1H")
    soxq_feat = compute_intraday_context(compute_technical_features(soxq))
    sox_show = [c for c in shared + ["rsi_14", "macd_hist", "bb_pctb", "atr_pct",
                                     "vol_zscore", "vwap_dist"] if c in soxq_feat.columns]
    sox_view = soxq_feat[sox_show].tail(250).copy()
    sox_view["Datetime"] = ny_str(sox_view["Datetime"])
    sox_table = table(sox_view, rename={
        "Datetime": "Fecha/hora (ET)", "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Volume": "Volumen"})

    sox_coef = sox_coefficients(dev, feats_d)
    sox_coef_table = table(sox_coef.head(18), rename={
        "variable": "Variable", "coef_BUY": "Coef BUY", "coef_HOLD": "Coef HOLD",
        "coef_SELL": "Coef SELL"}, formatters={
        "Coef BUY": lambda v: f"{v:+.3f}", "Coef HOLD": lambda v: f"{v:+.3f}",
        "Coef SELL": lambda v: f"{v:+.3f}"})

    # ---------------- export CSV (hora de Nueva York) ----------------
    data_dir = WEB_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in price_cols + ind_cols if c in feats_df.columns]
    export = feats_df[export_cols].copy()
    export["Datetime"] = ny_str(export["Datetime"])
    export.to_csv(data_dir / "nvda_features_1h.csv", index=False)

    sox_export = soxq_feat[sox_show].copy()
    sox_export["Datetime"] = ny_str(sox_export["Datetime"])
    sox_export.to_csv(data_dir / "soxq_features_1h.csv", index=False)

    lab_export = pd.read_csv(LABELED_CSV)
    for c in ["Datetime", "label_start", "label_end"]:
        if c in lab_export.columns:
            lab_export[c] = ny_str(lab_export[c])
    lab_export.to_csv(data_dir / "nvda_labeled_1h.csv", index=False)

    news_export = news[news_cols].copy()
    news_export["published_utc"] = ny_str(news_export["published_utc"])
    news_export = news_export.rename(columns={"published_utc": "fecha_hora_et"})
    news_export.to_csv(data_dir / "news_impact.csv", index=False)

    pred_out = pd.DataFrame({
        "datetime_et": ny_str(test["Datetime"]),
        "close": test["adj_close"].round(3).to_numpy(),
        "label_real": labels_test,
        "future_ret": test["future_ret"].round(5).to_numpy(),
        "p_BUY": proba_test[:, 0].round(4),
        "p_HOLD": proba_test[:, 1].round(4),
        "p_SELL": proba_test[:, 2].round(4),
        "signal": np.where(sig_labeled == 1, "BUY",
                           np.where(sig_labeled == -1, "SELL", "HOLD")),
    })
    pred_out.to_csv(data_dir / "predictions_holdout.csv", index=False)
    bt_out = pd.DataFrame({
        "datetime_et": ny_str(bars_test["Datetime"]),
        "close": bars_test["adj_close"].round(3).to_numpy(),
        "signal": np.where(sig_bars == 1, "BUY", np.where(sig_bars == -1, "SELL", "HOLD")),
        "position_long": bt_long["positions"],
        "return_long": bt_long["returns"].round(6),
        "equity_long": bt_long["equity"].round(5),
        "position_ls": bt_ls["positions"],
        "return_ls": bt_ls["returns"].round(6),
        "equity_ls": bt_ls["equity"].round(5),
        "buy_hold_equity": bt_bh["equity"].round(5),
    })
    bt_out.to_csv(data_dir / "backtest_holdout.csv", index=False)

    # ---------------- assets de graficos (fechas en ET) ----------------
    chart_dates = ny_str(bars_test["Datetime"]).tolist()
    dd = bt_long["equity"] / np.maximum.accumulate(bt_long["equity"]) - 1.0
    vol_ratio = (feats_df["Volume"] / feats_df["Volume"].rolling(20).mean().shift(1))
    candle = feats_df.tail(1200).copy()
    candle["vol_ratio"] = vol_ratio.tail(1200)
    active_long = bt_long["returns"][bt_long["positions"] != 0]
    sent_year_assets = {
        "years": sent_year["year_ny"].astype(int).tolist(),
        "avg": sent_year["avg_sent"].round(4).tolist(),
        "n": sent_year["n"].astype(int).tolist(),
    }
    daily = feats_df.groupby("date_ny", sort=True).agg(
        adj_close=("adj_close", "last"), sox_close=("sox_close", "last")).reset_index()
    daily = daily.tail(180)
    sox_daily = {
        "dates": [str(d) for d in daily["date_ny"]],
        "nvda": (daily["adj_close"] / daily["adj_close"].iloc[0]).round(4).tolist(),
        "sox": (daily["sox_close"] / daily["sox_close"].iloc[0]).round(4).tolist(),
    }
    sox_roll = {
        "dates": ny_str(feats_df.tail(252)["Datetime"]).tolist(),
        "beta": feats_df.tail(252)["beta_60b"].round(3).tolist(),
        "corr": feats_df.tail(252)["corr_60b"].round(3).tolist(),
        "ratio_z": feats_df.tail(252)["sox_ratio_z"].round(3).tolist(),
    }

    assets = {
        "kpi": {
            "H": DAYTRADE_HORIZON_BARS,
            "tau": DAYTRADE_TAU_PCT,
            "threshold": SIGNAL_PROBA_THRESHOLD,
            "cost": COST_CONSERVATIVE,
            "data_start": DATA_START_DATE,
            "test_start": set_d["test_start"],
            "n_features": set_d["n_features"],
            "model": "Logistic Regression (NVDA + SOXQ + noticias sin conteos)",
            "generated": pd.Timestamp.now("UTC").tz_convert(NY_TZ).strftime("%Y-%m-%d %H:%M ET"),
        },
        "equity": {
            "dates": chart_dates,
            "long": bt_long["equity"].round(5).tolist(),
            "ls": bt_ls["equity"].round(5).tolist(),
            "bh": bt_bh["equity"].round(5).tolist(),
            "drawdown": dd.round(5).tolist(),
        },
        "candles": {
            "dates": ny_str(candle["Datetime"]).tolist(),
            "open": candle["Open"].round(2).tolist(),
            "high": candle["High"].round(2).tolist(),
            "low": candle["Low"].round(2).tolist(),
            "close": candle["Close"].round(2).tolist(),
            "vol_ratio": candle["vol_ratio"].round(3).tolist(),
            "up": (candle["Close"] >= candle["Open"]).astype(int).tolist(),
            "vwap": candle["vwap"].round(2).tolist(),
            "sma20": candle["sma_20"].round(2).tolist(),
            "sma50": candle["sma_50"].round(2).tolist(),
        },
        "news_bar": news_matrix,
        "sent_year": sent_year_assets,
        "sox_daily": sox_daily,
        "sox_roll": sox_roll,
        "bt7": bt7,
    }

    # ---------------- HTML ----------------
    data = {
        "KPI_HEADER": (f"H={DAYTRADE_HORIZON_BARS} · τ={DAYTRADE_TAU_PCT}% · "
                       f"{set_d['n_features']} variables · test desde {set_d['test_start']}"),
        "CARD_THRESHOLD": f"{SIGNAL_PROBA_THRESHOLD:.0%}",
        "HORIZON": str(DAYTRADE_HORIZON_BARS),
        "TAU_PCT": f"{DAYTRADE_TAU_PCT:g}",
        "BT7_TABLE": bt7_table,
        "SOX_BETA": f"{last_row['beta_60b']:.2f}",
        "SOX_CORR": f"{last_row['corr_60b']:.2f}",
        "SOX_RATIO_Z": f"{last_row['sox_ratio_z']:+.2f}",
        "SOX_ALPHA": pct(last_row["alpha_1b"]),
        "SIGNAL_CLASS": sig_class,
        "SIGNAL_LABEL": live_signal,
        "SIGNAL_PRICE": f"{last_row['Close']:.2f}",
        "SIGNAL_DATETIME": pd.to_datetime(last_dt, utc=True).tz_convert(NY_TZ).strftime("%Y-%m-%d %H:%M ET"),
        "PROB_BUY": pct(p_live[0]), "PROB_HOLD": pct(p_live[1]), "PROB_SELL": pct(p_live[2]),
        "SUCCESS_PROB": pct(success_p), "SUCCESS_HIST": pct(success_hist),
        "RISK_LEVEL": risk, "RISK_PCTILE": pct(risk_pctile), "RISK_TONE": risk_tone,
        "STATE": state, "STATE_TONE": state_tone, "STATE_SCORE": f"{state_score:+d}",
        "STATE_REASONS": " · ".join(state_reasons),
        "NEWS_TOTAL": f"{len(news_hist):,}",
        "NEWS_30D": f"{len(news_30d):,}",
        "SENT_24H": f"{sent_avg_24h:+.3f}", "SENT_7D": f"{sent_avg_7d:+.3f}",
        "SENT_TONE": ("tone-buy" if sent_avg_24h > 0.15 else
                      "tone-sell" if sent_avg_24h < -0.15 else "tone-hold"),
        "ATR_PCT": pct(last_row["atr_pct"]),
        "RSI_VALUE": f"{last_row['rsi_14']:.1f}",
        "CARD_OOF_F1": f"{set_d['oof']['f1_macro']:.3f}",
        "CARD_OOF_AUC": f"{set_d['oof'].get('roc_auc_ovr', float('nan')):.3f}",
        "CARD_HOLD_F1": f"{set_d['holdout']['f1_macro']:.3f}",
        "CARD_HOLD_AUC": f"{set_d['holdout'].get('roc_auc_ovr', float('nan')):.3f}",
        "CARD_BASELINE": "0.182",
        "CARD_ACC": f"{set_d['holdout']['accuracy']:.3f}",
        "CARD_BALACC": f"{set_d['holdout']['balanced_accuracy']:.3f}",
        "CARD_AUC_STD": f"±{set_d['oof'].get('auc_folds_std', float('nan')):.3f}",
        "CARD_NTEST": f"{set_d['n_test']:,}",
        "CARD_BARS": f"{len(bars_test):,}",
        "CARD_SESSIONS": f"{len(sessions):,}",
        "KPI_MODEL_ROWS": kpi_model_rows,
        "ABLATION_ROWS": ablation_rows,
        "CM_OOF": confusion_html(set_d["oof"]["confusion_matrix"]),
        "CM_HOLD": confusion_html(set_d["holdout"]["confusion_matrix"]),
        "KEY_IND_ROWS": key_ind_rows,
        "FACTOR_ROWS": factor_rows,
        "NVDA_TABLE": nvda_table,
        "SOX_TABLE": sox_table,
        "POS_NEWS_TABLE": pos_table,
        "NEG_NEWS_TABLE": neg_table,
        "RECENT_NEWS_TABLE": recent_news_table,
        "SOX_COEF_TABLE": sox_coef_table,
        "COST_SENS_ROWS": cost_sens_rows,
        "FIN_LONG_SHARPE": f"{bt_long['sharpe']:.2f}",
        "FIN_LONG_SORTINO": f"{bt_long['sortino']:.2f}",
        "FIN_LONG_MDD": pct(bt_long["max_dd"]),
        "FIN_LONG_CALMAR": f"{bt_long['calmar']:.2f}",
        "FIN_LONG_WIN": pct(bt_long["win_rate"]),
        "FIN_LONG_PF": f"{bt_long['profit_factor']:.2f}",
        "FIN_LONG_RET": pct(bt_long["total_return"]),
        "FIN_LONG_ANN": pct(bt_long["annual_return"]),
        "FIN_LONG_TRADES": f"{bt_long['n_trades']:,}",
        "FIN_LONG_COST": pct(bt_long["total_costs"], 2),
        "FIN_LONG_AVG": pct(bt_long["avg_trade"], 3),
        "FIN_LS_SHARPE": f"{bt_ls['sharpe']:.2f}",
        "FIN_LS_SORTINO": f"{bt_ls['sortino']:.2f}",
        "FIN_LS_MDD": pct(bt_ls["max_dd"]),
        "FIN_LS_CALMAR": f"{bt_ls['calmar']:.2f}",
        "FIN_LS_WIN": pct(bt_ls["win_rate"]),
        "FIN_LS_PF": f"{bt_ls['profit_factor']:.2f}",
        "FIN_LS_RET": pct(bt_ls["total_return"]),
        "FIN_LS_TRADES": f"{bt_ls['n_trades']:,}",
        "FIN_BH_RET": pct(bt_bh["total_return"]),
        "FIN_BH_SHARPE": f"{bt_bh['sharpe']:.2f}",
        "FIN_BH_MDD": pct(bt_bh["max_dd"]),
        "FIN_IC": f"{ic:.3f}",
        "FIN_TSTAT": f"{bt_long['tstat']:.2f}",
        "FIN_ACTIVE_BARS": f"{bt_long['active_bars']:,}",
        "BT_RAW_SHARPE": f"{bt_raw['sharpe']:.2f}",
        "BT_RAW_RET": pct(bt_raw["total_return"]),
        "COST_BPS": f"{COST_CONSERVATIVE * 100:.1f}",
        "COST_REALISTA": f"{COST_REALISTA * 100:.2f}",
        "FIN_REAL_SHARPE": f"{bt_long_real['sharpe']:.2f}",
        "FIN_REAL_RET": pct(bt_long_real["total_return"]),
        "FIN_REAL_COST": pct(bt_long_real["total_costs"], 2),
        "TEST_PERIOD": (f"{pd.to_datetime(bars_test['Datetime'].iloc[0], utc=True).date()} → "
                        f"{pd.to_datetime(bars_test['Datetime'].iloc[-1], utc=True).date()}"),
    }
    html_out = render(data, assets)
    out_path = WEB_DIR / WEB_HTML_NAME
    out_path.write_text(html_out, encoding="utf-8")
    logger.info(f"Informe generado: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
