# Quant Terminal

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Datos](https://img.shields.io/badge/datos-yfinance-green)
![Dashboard](https://img.shields.io/badge/dashboard-GitHub%20Pages-orange)

Un terminal cuantitativo personal: descarga datos diarios, calcula métricas y señales de trading sistemático, manda alertas a Telegram y lleva el seguimiento de cada estrategia en un dashboard público que se actualiza solo cada noche.

**[Ver el dashboard en vivo](https://davizz-dev13.github.io/quant-terminal/)**


## Qué hace

- **Señales diarias** sobre un universo de ETFs, futuros, semiconductores y Bitcoin: SPY, QQQ, GLD, GC=F, CL=F, TLT, AMD, TSM, ASML, AVGO y BTC-USD.
- **Alertas a Telegram** solo cuando algo cambia de verdad: si una estrategia entra en largo o sale de un activo, llega un mensaje. Si no hay cambio de régimen, no hay ruido.
- **Backtesting honesto**: las señales se ejecutan desde la vela siguiente, con costes de transacción y remuneración del cash (configurables en `config/settings.yaml`), para no inflar los resultados.
- **Dashboard autoactualizado**: curvas de rentabilidad por estrategia vs buy & hold, ganancias YTD, Sharpe y drawdown, y las últimas señales. Se regenera cada día laborable tras el cierre de EE.UU. con una GitHub Action.

## Estrategias

Siete estrategias de régimen (largo / fuera), todas sobre velas diarias:

| Estrategia | Idea |
|---|---|
| `SMA200` | Largo por encima de la media de 200 días |
| `EMA50` | Largo por encima de la EMA de 50 días |
| `EMA20>EMA50` | Cruce de medias exponenciales |
| `EMA10>EMA30` | Cruce rápido de medias exponenciales |
| `Donchian20` | Rotura del canal de Donchian de 20 días |
| `ROC60` | Momentum positivo a 60 días |
| `MeanReversionZ` | Compra caídas fuertes (z-score ≤ -2.25) con tendencia de fondo alcista |

`core/signals.py` incluye además un laboratorio de señales más amplio (ROC a 120/252 días, canales de Donchian de 50/100/200, filtros de volumen, RSI, ADX, ATR y dos estrategias de vol-target) para validar qué familias merecen más investigación antes de tocar nada en vivo.

## Instalación y uso

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Alertas de Telegram

1. Crea un bot con [@BotFather](https://t.me/BotFather) y copia el token y tu chat id en `.env` (plantilla en `.env.example`).
2. Prueba en seco: `python run_alerts.py --dry-run` (imprime las alertas sin enviar nada).
3. En vivo: `python run_alerts.py`. Programa una ejecución diaria tras el cierre con cron o el Programador de tareas.

Las alertas ya enviadas se deduplican en `data/alert_state.json` y cada corrida revisa las últimas sesiones, así que un día perdido no pierde una señal reciente.

### Validación

```bash
python validate_signals.py
python scripts/walk_forward.py
```

`validate_signals.py` descarga 10 años de datos, aplica costes y escribe `data/validation_results.csv`. Un Sharpe histórico positivo no basta: mira el comportamiento por activo, subperiodo y costes antes de fiarte de una señal.

`walk_forward.py` hace la validación seria: evalúa cada estrategia en pliegues anuales sucesivos usando solo el pasado disponible hasta cada año (out-of-sample) y reporta Sharpe mediano por año, porcentaje de años positivos y agregados OOS. Es lo más parecido a haberla operado de verdad.


### Estrategia Mag7 + Oro + BTC (SMA200 semanal)

Apartado propio en el dashboard: **[ver](https://davizz-dev13.github.io/quant-terminal/mag7/)**. Estrategia contrarian en timeframe semanal sobre las 7 magníficas, GLD y BTC-USD:

- **Entrada**: el cierre semanal cruza a la baja la SMA200 semanal.
- **Salida**: cuando el RSI(14) semanal supera 70 la posición queda armada; se vende al caer más de un 5% desde el máximo cierre posterior.
- Señal al cierre de la semana, ejecutable la semana siguiente. Las alertas nocturnas de Telegram también cubren estas señales semanales.

El dashboard separa siempre tres cosas: el **paper trading en vivo** (track record real desde el go-live, el número que importa), la **validación walk-forward** (robustez histórica) y el **backtest** (referencia, claramente etiquetado).

## Estructura

```
app/streamlit_app.py   Terminal visual (Streamlit)
core/                  Datos, métricas, señales y backtest
config/settings.yaml   Universo, confirmaciones, costes y alertas
scripts/               Generador de datos del dashboard
docs/                  Dashboard estático (GitHub Pages)
```

## Aviso

Esto es un proyecto de aprendizaje y seguimiento personal, no asesoramiento financiero. Las rentabilidades pasadas no garantizan resultados futuros.
