# BTC Reversal Intelligence v1

A local research prototype for BTCUSDT USDⓈ-M perpetual reversal strategy backtesting.

## Safety
- This project does **not** place live orders.
- Do not add API keys or trading permissions to this prototype.
- Results are research outputs, not evidence of future profitability.
- Historical order-book replay, exact funding-fee accounting, and robust execution simulation are not implemented yet.

## Run locally
Requires Python 3.10+.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit in a desktop or mobile browser on the same network/device setup.

## Current logic
- Instrument: BTCUSDT perpetual public market data.
- 1h: simple EMA slope regime filter.
- 15m: RSI + Bollinger-band extension + prior local high/low candidate.
- 5m: short-term swing-boundary confirmation.
- Entry: next candle open, with configurable fee/slippage assumptions.
- Risk: fixed equity fraction, stop-based position sizing, notional capped at 1x equity.
- Bar ambiguity: if stop and target both touch in the same candle, stop is assumed first.

## Important limitations
1. The strategy rules are preliminary and unvalidated.
2. Current backtest uses a simplified bar-based fill model.
3. Funding payments are omitted; the UI should not be interpreted as an all-in-cost backtest.
4. Order book / depth / trade-flow features are not included yet. Those require storing synchronized live market data or a suitable historical data source.
5. The higher-timeframe merge and signal rules should be audited before relying on any reported performance.
6. Do not use this as financial advice or as a live trading bot.
