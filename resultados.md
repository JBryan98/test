# Resultados — PRECIARIO_NVDA (backup)

> Fecha del análisis: **2026-09-20**
> Datos: `core/output/intradia/NVDA_intraday.csv`, `SOXQ_intraday.csv`, `core/output/news_history/news.csv`
> Corte temporal aplicado: **2021-06-11** (inicio de SOXQ; descarta pandemia, según `instrucciones-inicio.md`)
> Modelo evaluado: **Logistic Regression** (scaler + `class_weight=balanced`), walk-forward purging (purge=3, embargo=7)
> Sin Docker/PostgreSQL (eliminado el 2026-09-20)

---

## 1. Datos expandidos

| Activo | Barras 1h | Rango | Días | Observaciones |
|---|---:|---|---:|---|
| NVDA | 9,231 | 2021-06-11 → 2026-09-17 | 1,315 | 7 barras/día RTH, precios ajustados por splits |
| SOXQ | 9,229 | 2021-06-11 → 2026-09-17 | 1,313 | primeros días con volumen bajo |
| news.csv | 41,743 | 2020-01 → 2026-09-18 | — | solo NVDA; densidad muy desigual por año |

- Filas válidas para target H=3 (misma sesión): **5,226** (antes ~956 con 12 meses).
- Distribución de la serie `ret_1b`: incluye gaps overnight (std 1.18%); intradía puro ≈ 0.64%.

## 2. Etiquetado: tau adaptativo vs tau fijo (H=3)

Grid real con datos expandidos (5,226 filas válidas):

| tau | BUY % | HOLD % | SELL % | OOF F1 | OOF AUC | TEST F1 | TEST AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive (config actual) | 33.9 | 37.0 | 29.2 | 0.354 | 0.525 | 0.306 | 0.539 |
| 0.5% | 33.0 | 38.4 | 28.6 | 0.386 | 0.561 | 0.255 | 0.548 |
| **0.6%** | 29.8 | 44.6 | 25.6 | 0.398 | 0.576 | 0.292 | 0.581 |
| **0.7%** | 26.8 | 50.1 | 23.1 | 0.401 | 0.595 | 0.304 | 0.590 |
| **0.8%** | 24.4 | 54.6 | 21.0 | 0.404 | 0.602 | 0.324 | 0.594 |

- El `tau` adaptativo actual (media 0.536%) es el **peor** esquema en datos expandidos.
- `fixed` 0.6-0.8% cumple la meta de balance (HOLD 40-50%, BUY/SELL ~25%) y sube el ROC-AUC de 0.53 a 0.59-0.60.
- Tau adaptativo calculado con vol intraday pura (~0.31%) empeoraba más: los movimientos <0.4% son ruido.

## 3. Horizontes (misma sesión, máx. H=6)

| Config | Balance (H/B/S %) | OOF AUC | TEST AUC | TEST F1 |
|---|---:|---:|---:|---:|
| H=3, tau 0.8% | 54.6 / 24.4 / 21.0 | 0.602 | 0.594 | 0.324 |
| **H=4, tau 0.8%** | **49.0 / 27.9 / 23.2** | **0.599** | **0.602** | **0.380** |
| H=4, tau 1.0% | 58.1 / 23.1 / 18.8 | 0.615 | 0.598 | 0.345 |
| H=5, tau 1.0% | 52.7 / 26.2 / 21.1 | 0.589 | 0.584 | 0.391 |

- `H=4 + tau 0.8%` es el mejor equilibrio: balance exacto pedido y ~0.60 de AUC en OOF **y** holdout.
- **`future_ret_5d` no es expresable en 1h con agrupación por sesión**: H≥7 produce 0 filas válidas. Para 5 días se requieren velas diarias o permitir overnight (contradice la regla anti-gap).

## 4. Estabilidad temporal

- OOF por año (H=3, tau 0.8%): 2023 AUC 0.589 · 2024 0.561 · 2025 0.645.
- Incluyendo 2021-2022 en la validación (min_train 0.3-0.4): AUC baja a 0.55-0.58 → la señal se concentra en 2023-2025.
- Walk-forward `expanding` vs `rolling` (últimas 1,000-2,000 barras): el expanding es igual o mejor; no se justifica ventana móvil.

## 5. Noticias (hallazgo crítico)

- Features point-in-time con half-life 12h ya implementadas (`core/news/alpha_vantage.py:157`); funcionan sin leakage.
- **Densidad del news.csv no homogénea** (por año):

| Año | Artículos NVDA |
|---|---:|
| 2020 | 249 |
| 2021 | 442 |
| 2022 | 578 |
| 2023 | 594 |
| 2024 | 1,141 |
| 2025 | 11,442 |
| 2026 | 27,297 |

- No hay URLs duplicadas (sí ~10% de títulos repetidos). La diferencia de densidad es real.
- **Causa raíz identificada:** las features de *conteo* (`cnt_w*`, `has_sentiment_24h`) y `hours_since_news` capturan la densidad del dataset, no la señal:

| Año | `sent_decay` | `sent_w1d` | `cnt_w1d` | `hours_since_news` |
|---|---:|---:|---:|---:|
| 2021 | 0.262 | 0.194 | 1.6 | 25.2 |
| 2023 | 0.231 | 0.191 | 2.1 | 18.1 |
| 2024 | 0.279 | 0.272 | 4.0 | 6.5 |
| 2025 | 0.231 | 0.236 | 37.5 | 3.6 |
| 2026 | 0.235 | 0.235 | 120.3 | 0.2 |

- El *nivel* de sentimiento (`sent_decay`, `sent_w1d`) es estable; el *volumen* de noticias crece 75×.
- Efecto medido (H=4, tau 0.8%):

| Feature set | OOF AUC (mf 0.5) | HOLDOUT AUC | HOLDOUT F1 |
|---|---:|---:|---:|
| B sin noticias | 0.5987 | 0.6019 | 0.3795 |
| B + TODAS (con `cnt_*`) | 0.6002 | **0.5074** | 0.3033 |
| **B + noticias sin `cnt_*`** | **0.5999** | **0.6012** | **0.3978** |
| B + solo `sent_decay`/`sent_momentum` | 0.5979 | 0.6011 | 0.3818 |

- **Conclusión: las noticias SÍ se pueden incluir, eliminando las columnas de conteo.** Mantener `sent_*` (ventanas ponderadas por relevancia), `sent_decay` y `sent_momentum`; excluir `cnt_*`, `has_sentiment_24h` y idealmente `hours_since_news`.
- Mejora pendiente recomendada: recalcular el sentimiento desde `title`+`summary` (ya presentes en `news.csv`) con FinBERT/VADER para homogeneizar calidad entre años, y/o re-descargar Alpha Vantage con protocolo uniforme. Para cobertura histórica homogénea: GDELT. Scraping (RSS Yahoo/Google) solo sirve para producción en vivo, no backfill.

## 6. Variables de sesión (vwap_dist, orb_break)

| Variable | IC Spearman vs future_ret (H=3) |
|---|---:|
| `dist_orb_low` | +0.039 |
| `above_vwap_frac` (nueva) | +0.038 |
| `cum_day_ret` | +0.036 |
| `orb_range_pct` (nueva) | +0.031 |
| `gap_open_pct` | +0.030 |
| `dist_orb_high` | +0.018 |
| `vwap_dist` | +0.014 |
| `orb_break_up` | +0.007 |
| `orb_break_down` | −0.013 |

- Añadirlas no mejora AUC (0.557 vs 0.561); quitarlas es neutro/levemente mejor (+0.003).
- `orb_break_up/down` son redundantes con `dist_orb_*` (mismo signo). Se recomienda reemplazarlas por eventos/normalizadas solo por limpieza.

## 7. Configuración final (confirmada 2026-09-20)

```dotenv
MODEL_TAU_MODE=fixed
MODEL_TAU_PCT=0.8
MODEL_HORIZON_BARS=4
MODEL_PURGE_BARS=4
MODEL_EMBARGO_BARS=7
MODEL_USE_PATTERNS=0
MODEL_USE_NEWS_COUNTS=0
SIGNAL_PROBA_THRESHOLD=0.40
```

- Nota umbral de señal: con el modelo de 48 variables las probabilidades rara vez superan
  0.55 (máx. holdout ≈ 0.69, mediana ≈ 0.48); con 0.55 **no se generaba ninguna operación**.
  Se ajustó a **0.40** (≈100 señales en 783 barras del holdout).
- Nota H=4: al exigir futuro dentro de la misma sesión, las últimas 4 barras del día no
  tienen etiqueta. El modelo se entrena con las **3 primeras barras diarias** (09:30-12:00 ET)
  y es donde su señal es válida.

- Feature set: **NVDA + SOXQ + noticias SIN columnas de conteo** (sin `pat_*`).
- Decisión de señal: no usar `argmax` solo; aplicar `SIGNAL_PROBA_THRESHOLD` (KPIS.md §12.2) porque el recall BUY/SELL a 0.5 es bajo (0.18/0.11 con H=4/tau 0.8%).

## 8. Pendientes

- [x] Confirmado H=4 / tau fijo 0.8% como target oficial (balance 49/28/23).
- [x] Noticias: excluir `cnt_*`/`has_sentiment*` del feature set (loss en holdout eliminada).
- [ ] Homogeneizar calidad del sentimiento (FinBERT sobre `title`+`summary`, GDELT o re-pull AV uniforme).
- [x] Integrar los CSV nuevos en `core/feature_engineering.py` (CSV expandido + corte 2021-06-11).
- [x] Reentrenar y guardar artefactos con la config elegida (modelo_nvda_logreg.pkl, metrics.json).
- [x] Informe HTML de 8 pestañas generado (`web/REPORT_NVDA/index.html`).
- [x] Eliminar Docker/PostgreSQL del proyecto operativo (2026-09-20).

## 9. Nota de reproducibilidad

Los análisis se hicieron con scripts de sonda temporales (no versionados) sobre los CSV crudos,
reutilizando `core/feature_engineering.py` y los mismos folds del trainer
(`purged_walk_forward`: 5 folds, purge=3, embargo=7, 80% dev / 20% holdout por sesiones).
No se modificó el dataset original.

---

## 10. Alternativas para las noticias (guardado para fase FinBERT/GDELT/AV/scraping)

Estado actual: se usa **solo Alpha Vantage** (`core/output/news_history/news.csv`), con el
sentimiento ya calculado por AV. Feature set válido: `sent_*` + `sent_decay` + `sent_momentum`,
**excluyendo `cnt_*`, `has_sentiment_24h` y (preferible) `hours_since_news`**.

### 10.1 FinBERT sobre `title` + `summary` (recomendado a corto plazo)

- **Qué:** recalcular el sentimiento localmente desde el texto que ya está en `news.csv`
  (`title`, `summary`). Modelo `ProsusAI/finbert` (HuggingFace) o `yiyanghkust/finbert-tone`.
- **Por qué:** elimina la dependencia del score precalculado de AV y homogeneiza la medición
  entre 2020 y 2026 (mismo léxico/reglas para todos los años). Corre offline, gratis.
- **Cómo:** batch de títulos+resúmenes → probabilidades {positive, negative, neutral} →
  score continuo `p_pos − p_neg`; reemplaza `sentiment_score` de AV manteniendo el mismo
  pipeline point-in-time (half-life 12h).
- **Coste:** ~42k artículos; en GPU RTX 5090 minutos; en CPU ~1-2 h. Requiere `transformers`,
  `torch` (no están en `requirements.txt`).
- **Riesgo:** FinBERT está entrenado en inglés financiero; los titulares de AV ya son en inglés.
  Validar contra una muestra etiquetada a mano.

### 10.2 GDELT (histórico homogéneo)

- **Qué:** base global de noticias con timestamps desde 2015. API DOC 2.0 (últimos 3 meses) +
  GDELT 2.0 GKG/Events históricos vía BigQuery o CSV públicos.
- **Por qué:** cobertura parecida en 2021-2026 (resuelve la densidad desigual del news.csv).
- **Cómo:** consultar por `"NVIDIA"`/`NVDA`, extraer fecha + titular/fuente, aplicar FinBERT o el
  `tone` de GDELT. Cargar como `published_utc, ticker, relevance_score, score, title, source`.
- **Riesgo:** mezclar fuentes GDELT+AV duplicaría noticias; decidir una sola fuente por período
  o deduplicar por título normalizado.

### 10.3 Re-pull uniforme de Alpha Vantage

- **Qué:** volver a descargar `NEWS_SENTIMENT` para 2021-06 → hoy con **protocolo idéntico**
  (mismos `time_from/time_to`, `limit`, `tickers=NVDA`).
- **Por qué:** el dataset actual se armó con protocolos distintos (2025-26 tiene 10-25× más
  artículos). Un re-pull mensual uniforme iguala la densidad.
- **Cómo:** chunks mensuales (`time_from=YYYYMM01T0000`, `time_to=...`), dedupe por URL/título.
  Free tier 25 req/día → ~64 requests = ~3 días; premium acelera.
- **Riesgo:** AV puede tener cobertura histórica intrínsecamente menor en años antiguos; el
  re-pull no garantiza densidad igual, solo protocolo igual.

### 10.4 Scraping RSS (Google News / Yahoo Finance)

- **Qué:** scrapear titulares + timestamp de RSS/HTML (Google News `q=NVDA`, Yahoo Finance
  NVDA news, Benzinga, PR Newswire).
- **Uso:** **producción en vivo** (bot horario), no backtest: los RSS solo exponen noticias
  recientes, no historia.
- **Cómo:** `feedparser`/`requests` + `BeautifulSoup`; aplicar FinBERT en tiempo real;
  almacenar en el mismo esquema `published_utc, ticker, source, title, score`.
- **Riesgo:** términos de uso de los sitios, paywalls y cambios de HTML; usar con moderación
  y cachear.

### 10.5 Decisión sugerida

1. **Ahora:** AV con `sent_*` sin conteos (ya validado, HOLDOUT AUC 0.601 / F1 0.398).
2. **Siguiente:** FinBERT sobre `title`+`summary` (misma tabla, sin cambiar el pipeline).
3. **Después:** GDELT para cubrir 2021-2024 con densidad homogénea (o re-pull AV).
4. **Producción:** RSS Google/Yahoo + FinBERT para el bot en vivo.

---

## 11. Informe HTML final (2026-09-20)

Generador: `scripts/report/build_report.py` (+ `report.css`, `report.js`, `report_body.html`).
Salida: `web/REPORT_NVDA/index.html` (~470 KB) y `web/REPORT_NVDA/data/*.csv`.

| Pestaña | Contenido |
|---|---|
| 01 KPIs Financieros | Backtest holdout: Sharpe, Sortino, Max DD, Calmar, Win Rate, Profit Factor, IC Spearman, curva de equity, drawdown, comparación con Buy & Hold |
| 02 KPIs del Modelo | OOF y HOLDOUT: F1 macro, ROC-AUC, precision/recall, matrices de confusión, ablaciones A/B/D |
| 03 Datos 1H | Tablas NVDA/SOXQ + botones de exportación CSV completos |
| 04 Impacto de Noticias | Impacto = relevancia × \|sentimiento\| en terciles (rojo alto / naranja medio / verde bajo), agrupadas en positivas y negativas + gráfico "Sentimiento Promedio por año" |
| 05 Dashboard Descriptivo | Estado de mercado, total de noticias, sentimiento promedio, gráfico de velas 1H (VWAP + SMA + volumen), tabla de noticias |
| 06 Dashboard Prescriptivo | Señal, probabilidad de éxito, precio actual, riesgo, backtest 7 días con dirección LONG/SHORT/FLAT, factores de recomendación e indicadores clave |
| 07 Auditoría | Glosario explicativo de todas las métricas |
| 08 SOXQ & NVDA | Beta/correlación/alpha, precios normalizados, ablación y coeficientes del modelo |

Plotly está **empaquetado localmente** (`web/REPORT_NVDA/assets/plotly-2.35.2.min.js`, con
fallback a CDN), así que los gráficos funcionan sin internet. Las fuentes siguen por CDN con
fallback del sistema. Los CSV de `data/` se descargan desde la pestaña 03, con definición de
cada dataset. El título del prescriptivo usa "NVIDIA" (corregido del "NVIDEA" original).

**Zona horaria:** el cálculo interno usa UTC (IBKR/Alpha Vantage); todas las tablas, gráficos y
CSV exportados muestran la hora de Nueva York (ET). `backtest_holdout.csv` ahora incluye las dos
estrategias (`position_long`/`return_long`/`equity_long` y `position_ls`/`return_ls`/`equity_ls`).

**Corrección importante del backtest (2026-09-20):** la primera versión usaba solo las filas
etiquetadas (3 por sesión); al encadenar sus retornos, cada fila abarcaba de golpe 5-7 horas y el
gap nocturno, inflando los KPIs. El backtest corregido usa **todas las barras horarias** del
holdout (1,827 barras, 261 sesiones), una entrada por sesión en la primera señal y salida a las
H barras o al cierre (sin overnight), con ejecución en t+1.

**Lectura de los KPIs financieros (corregidos):**

| Estrategia (coste 0.10%) | Sharpe | Retorno | Max DD | Trades | Costes |
|---|---:|---:|---:|---:|---:|
| Day trading long-only | 0.10 | 0.5% | −8.2% | 41 | 8.2% |
| Day trading long/short (hipotética) | 0.22 | 2.0% | −12.8% | 71 | 14.2% |
| Argmax sin umbral (referencia) | −0.25 | −3.4% | — | — | — |
| Buy & Hold (mismas barras) | 0.80 | 27.8% | −22.1% | 1 | — |

- **Con el coste conservador del proyecto (0.10%) la señal no sobrevive**: el coste acumulado
  (8.2%) supera el retorno (0.5%) y el t-stat es 0.10.
- **Sensibilidad de costes** (long-only): 0.01% → Sharpe 0.84 / retorno 8.2%; 0.02% → Sharpe 0.76 /
  retorno 7.3% / costes 1.64%; 0.05% → Sharpe 0.51; 0.10% → Sharpe 0.10. Referencia realista para
  NVDA (muy líquido): ~0.02%.
- **Ningún escenario alcanza significancia estadística** (t < 2). El Sharpe del benchmark (0.80)
  es comparable al de la estrategia con coste realista (0.76), pero la estrategia tiene mejor
  drawdown (−8% vs −22%) porque está en efectivo ~91% de las barras.
- Conclusión honesta: la clasificación tiene señal modesta (AUC 0.60) pero **no se traduce en
  rentabilidad neta robusta** con la configuración actual; el alto coste relativo y el bajo
  número de operaciones por sesión son el cuello de botella.

## 12. Especificación de extracción para PostgreSQL (pendiente de aprobación)

Tablas propuestas (nombres y columnas a confirmar antes de crearlas):

1. `fact_bars_1h`: `datetime` (PK), `symbol`, OHLCV, `ret_1b`, `log_ret_1b`, `ret_3b`, `ret_1d`,
   `sma_20`, `sma_50`, `dist_sma_20`, `dist_sma_50`, `macd`, `macd_signal`, `macd_hist`,
   `macd_pct`, `macd_hist_pct`, `rsi_14`, `bb_mid`, `bb_pctb`, `bb_width`, `atr_14`, `atr_pct`,
   `vol_zscore`, `volume_surge`, `hl_range_pct`, `close_pos`, `bar_of_day`, `bars_left`,
   `is_first_bar`, `is_last_bar`, `hour_ny`, `minute_ny`, `day_of_week`, `vwap`, `vwap_dist`,
   `orb_high`, `orb_low`, `dist_orb_high`, `dist_orb_low`, `orb_break_up`, `orb_break_down`,
   `cum_day_ret`, `gap_open_pct`, `sox_ret_1b`, `sox_ret_1d`, `sox_ratio`, `sox_ratio_z`,
   `beta_60b`, `corr_60b`, `alpha_1b`, `alpha_5b`, `rel_strength_1d`, `sent_decay`, `sent_w6h`,
   `sent_w1d`, `sent_w3d`, `sent_w7d`, `sent_w30d`, `sent_momentum`, `hours_since_news`.
   (Sin `cnt_*` ni `pat_*`.)
2. `fact_labels`: `datetime` (FK), `future_ret`, `tau`, `label`, `label_horizon_bars`,
   `label_start`, `label_end`.
3. `fact_predictions`: `datetime`, `modelo`, `split` (OOF/HOLDOUT/LIVE), `signal`,
   `p_buy`, `p_hold`, `p_sell`, `threshold`.
4. `fact_backtest`: `datetime`, `estrategia`, `position`, `strategy_return`, `equity`,
   `buy_hold_equity`, `drawdown`.
5. `fact_news`: `published_utc`, `ticker`, `source`, `title`, `sentiment_label`,
   `sentiment_score`, `relevance_score`, `impact_score`, `impact_level`, `polarity`.
6. `kpi_model`: `modelo`, `feature_set`, `split`, `f1_macro`, `roc_auc`, `accuracy`,
   `balanced_accuracy`, `precision_buy`, `recall_buy`, `precision_sell`, `recall_sell`, `n_eval`.
7. `kpi_financial`: `estrategia`, `fecha_inicio`, `fecha_fin`, `sharpe`, `sortino`, `max_dd`,
   `calmar`, `win_rate`, `profit_factor`, `ic_spearman`, `total_return`, `annual_return`,
   `n_trades`, `total_costs`, `buyhold_return`.

Origen: `web/REPORT_NVDA/data/*.csv` (mismos datos que alimentan el informe).
Pendiente: confirmar si se carga también `fact_news` completo (42k filas) o solo noticias
con `relevancia >= 0.3` y deduplicadas por título.

---

## 13. Actualización de datos — 2026-09-22

Llegaron los 3 CSV con datos del día: NVDA (+21 barras), SOXQ (+21 barras) y news.csv (+74 noticias,
hasta 2026-09-22 17:34 UTC). Se eliminaron metadatos `:Zone.Identifier` de Windows.

Pipeline re-ejecutado completo:

| Etapa | Resultado |
|---|---|
| Features | 9,190 filas × 75 cols |
| Labels (H=4, tau 0.8%) | 3,920 filas — BUY 27.9% / HOLD 49.0% / SELL 23.2% |
| Modelo reentrenado (OOF) | F1 0.4034 · ROC-AUC 0.5980 · n=1,960 |
| Informe: OOF (dev) | F1 0.4026 · AUC 0.5978 |
| Informe: HOLDOUT | F1 0.4044 · AUC **0.6009** |
| Señal en vivo | **HOLD** (BUY 27.4% / HOLD 52.1% / SELL 20.5%) — barra 2026-09-22 14:00 ET, precio 229.31 |
| Estado / riesgo | BULLISH / BAJO |

KPIs financieros corregidos (coste conservador 0.10%): long-only Sharpe 0.28, retorno 2.4%,
MDD −8.2%, 42 operaciones, costes 8.4%, t-stat 0.29; L/S Sharpe 0.53, retorno 6.3%, MDD −12.1%.
Se mantiene la conclusión: sin significancia estadística y muy sensible a costes.

**Rigor añadido:** se excluye la barra en curso (sin cerrar) tanto del etiquetado
(`core/labels.py`) como del estado en vivo del informe (`scripts/report/build_report.py`),
para no usar datos parciales de la sesión de hoy como "futuro".
