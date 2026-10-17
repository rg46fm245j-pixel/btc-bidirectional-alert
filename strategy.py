import time
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://fapi.binance.com"

def fetch_klines(days=30, symbol="BTCUSDT", interval="5m"):
    """Fetch public USDⓈ-M Futures candles. No API key or trading permission required."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            f"{BASE_URL}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cursor,
                    "endTime": end_ms, "limit": 1500},
            timeout=20
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1500:
            break
        time.sleep(0.15)

    if not rows:
        raise RuntimeError("没有获取到K线数据。")
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    for c in ["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # Exclude the currently forming candle.
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") < now_ms].copy()
    return df.reset_index(drop=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    x = df.copy().set_index("time").sort_index()
    x15 = x.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    x1h = x.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    x15["rsi15"] = rsi(x15["close"], 14)
    x15["ma20"] = x15["close"].rolling(20).mean()
    x15["std20"] = x15["close"].rolling(20).std(ddof=0)
    x15["bb_lower"] = x15["ma20"] - 2*x15["std20"]
    x15["bb_upper"] = x15["ma20"] + 2*x15["std20"]
    # Use prior bars only to define the candidate zone.
    x15["prior_low"] = x15["low"].rolling(20).min().shift(1)
    x15["prior_high"] = x15["high"].rolling(20).max().shift(1)

    x1h["ema50"] = x1h["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    x1h["ema_slope"] = x1h["ema50"].pct_change(3)
    # Merge only completed higher-timeframe bars into each 5m timestamp.
    x = x.reset_index()
    x = pd.merge_asof(x.sort_values("time"), x15[["rsi15","bb_lower","bb_upper","prior_low","prior_high"]].reset_index().sort_values("time"),
                      on="time", direction="backward", allow_exact_matches=True)
    x = pd.merge_asof(x.sort_values("time"), x1h[["ema50","ema_slope"]].reset_index().sort_values("time"),
                      on="time", direction="backward", allow_exact_matches=False)
    x["rsi5"] = rsi(x["close"], 14)
    x["recent_low"] = x["low"].rolling(12).min().shift(1)
    x["recent_high"] = x["high"].rolling(12).max().shift(1)
    # Simple reversal confirmation: close crosses the prior short-term swing boundary.
    x["long_confirm"] = (x["close"] > x["recent_high"]) & (x["close"].shift(1) <= x["recent_high"].shift(1))
    x["short_confirm"] = (x["close"] < x["recent_low"]) & (x["close"].shift(1) >= x["recent_low"].shift(1))
    # These are research filters, not validated predictive claims.
    x["long_candidate"] = (x["rsi15"] < 30) & (x["close"] < x["bb_lower"]) & (x["low"] <= x["prior_low"])
    x["short_candidate"] = (x["rsi15"] > 70) & (x["close"] > x["bb_upper"]) & (x["high"] >= x["prior_high"])
    # Avoid countertrend entries when hourly EMA slope is exceptionally strong.
    x["regime_ok"] = x["ema_slope"].abs() < 0.025
    return x.reset_index(drop=True)

def run_backtest(df, initial_capital=10000.0, risk_pct=0.005, fee_bps=5.0,
                 slippage_bps=2.0, min_rr=2.0):
    x = add_indicators(df)
    equity = float(initial_capital)
    peak = equity
    max_dd = 0.0
    trades = []
    curve = []
    position = None
    fee_rate = fee_bps / 10000
    slip_rate = slippage_bps / 10000
    cooldown_until = -1

    for i in range(100, len(x)-1):
        row = x.iloc[i]
        nxt = x.iloc[i+1]
        curve.append({"time": row["time"], "equity": equity})
        if position:
            exit_price = None
            reason = None
            # Conservative bar-level handling: if stop and target both hit, stop wins.
            if position["side"] == "LONG":
                if row["low"] <= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif row["high"] >= position["target"]:
                    exit_price, reason = position["target"], "target"
            else:
                if row["high"] >= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif row["low"] <= position["target"]:
                    exit_price, reason = position["target"], "target"
            if exit_price is not None:
                exit_exec = exit_price * (1 - slip_rate if position["side"] == "LONG" else 1 + slip_rate)
                gross = (exit_exec - position["entry_exec"]) * position["qty"] * (1 if position["side"]=="LONG" else -1)
                fees = (position["entry_exec"] + exit_exec) * position["qty"] * fee_rate
                net = gross - fees
                equity += net
                trades.append({**position, "exit_time": row["time"], "exit_price": exit_price,
                               "reason": reason, "gross_pnl": gross, "fees_est": fees,
                               "net_pnl": net, "equity_after": equity})
                position = None
                cooldown_until = i + 2
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak-equity)/peak if peak else 0)
                continue

        if position or i < cooldown_until:
            continue
        if not bool(row.get("regime_ok", False)):
            continue

        side = None
        if bool(row.get("long_candidate", False)) and bool(row.get("long_confirm", False)):
            side = "LONG"
        elif bool(row.get("short_candidate", False)) and bool(row.get("short_confirm", False)):
            side = "SHORT"
        if not side:
            continue

        # Enter at next candle open to avoid using the signal close as a guaranteed fill.
        entry = float(nxt["open"])
        if side == "LONG":
            stop = float(min(row["low"], row["recent_low"]) * (1 - 0.001))
            risk_per_unit = entry - stop
            target = entry + min_rr * risk_per_unit
        else:
            stop = float(max(row["high"], row["recent_high"]) * (1 + 0.001))
            risk_per_unit = stop - entry
            target = entry - min_rr * risk_per_unit
        if not np.isfinite(risk_per_unit) or risk_per_unit <= 0 or target <= 0:
            continue
        # Ensure gross target is not too close to costs.
        cost_per_unit = entry * (2*fee_rate + 2*slip_rate)
        if min_rr * risk_per_unit <= cost_per_unit:
            continue
        risk_cash = equity * risk_pct
        qty = risk_cash / risk_per_unit
        # Avoid extreme notional/leverage in this research prototype: max 1x equity notional.
        qty = min(qty, equity / entry)
        entry_exec = entry * (1 + slip_rate if side == "LONG" else 1 - slip_rate)
        position = {"side": side, "entry_time": nxt["time"], "entry_price": entry,
                    "entry_exec": entry_exec, "stop": stop, "target": target,
                    "qty": qty, "risk_budget": risk_cash, "signal_time": row["time"]}

    # Close any remaining position at final close, labelled as end-of-data.
    if position and len(x):
        last = x.iloc[-1]
        exit_price = float(last["close"])
        exit_exec = exit_price * (1 - slip_rate if position["side"] == "LONG" else 1 + slip_rate)
        gross = (exit_exec-position["entry_exec"])*position["qty"]*(1 if position["side"]=="LONG" else -1)
        fees = (position["entry_exec"]+exit_exec)*position["qty"]*fee_rate
        net = gross-fees
        equity += net
        trades.append({**position, "exit_time": last["time"], "exit_price": exit_price,
                       "reason":"end_of_data", "gross_pnl":gross, "fees_est":fees,
                       "net_pnl":net, "equity_after":equity})

    tdf = pd.DataFrame(trades)
    if len(tdf):
        wins = tdf[tdf["net_pnl"] > 0]["net_pnl"].sum()
        losses = -tdf[tdf["net_pnl"] < 0]["net_pnl"].sum()
        pf = float(wins/losses) if losses > 0 else (None if wins == 0 else float("inf"))
        win_rate = float((tdf["net_pnl"] > 0).mean()*100)
        avg_trade = float(tdf["net_pnl"].mean())
    else:
        pf, win_rate, avg_trade = None, 0.0, 0.0
    metrics = {
        "trades": int(len(tdf)),
        "net_pnl": float(equity-initial_capital),
        "return_pct": float((equity/initial_capital-1)*100),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_trade": avg_trade,
        "max_drawdown_pct": float(max_dd*100),
        "final_equity": float(equity)
    }
    return {"metrics":metrics, "trades":trades, "equity":pd.DataFrame(curve)}
