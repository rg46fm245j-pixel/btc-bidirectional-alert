import time
import requests
import pandas as pd
import numpy as np


KRAKEN_URL = "https://futures.kraken.com/api/charts/v1/trade/PF_XBTUSD"


def fetch_klines(days=30, symbol="BTCUSDT", interval="5m"):
    """获取 Kraken PF_XBTUSD 期货 K 线；不是 Binance BTCUSDT 行情。"""
    interval_ms_map = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
    }

    if interval not in interval_ms_map:
        raise ValueError("仅支持 5m、15m 和 1h 周期。")

    interval_ms = interval_ms_map[interval]
    now_ms = int(time.time() * 1000)
    start_sec = (now_ms - int(days * 86400000)) // 1000
    end_sec = now_ms // 1000
    cursor_sec = start_sec
    rows = []

    while cursor_sec < end_sec:
        response = requests.get(
            f"{KRAKEN_URL}/{interval}",
            params={
                "from": cursor_sec,
                "to": end_sec,
                "count": 500,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("candles", [])

        if not batch:
            break

        rows.extend(batch)
        last_ms = int(batch[-1]["time"])
        next_cursor = (last_ms + interval_ms) // 1000

        if next_cursor <= cursor_sec:
            break

        cursor_sec = next_cursor

        if not payload.get("more_candles", False):
            break

        time.sleep(0.15)

    if not rows:
        raise RuntimeError("Kraken 没有返回 K 线数据，请稍后重试。")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("time").sort_values("time").copy()
    df["open_time"] = pd.to_numeric(
        df["time"], errors="coerce"
    )

    df = df.dropna(subset=["open_time"])
    df["open_time"] = df["open_time"].astype("int64")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )
    df["close_time"] = df["open_time"] + interval_ms - 1
    df = df[df["close_time"] < now_ms].copy()

    for col in [
        "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]:
        df[col] = 0

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )

    return df.reset_index(drop=True)


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    # 持续上涨时 RSI 为 100，而不是空值。
    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0), 100.0
    )
    return result


def add_indicators(df):
    if df.empty:
        raise ValueError("K 线数据为空。")

    x = df.copy()
    x["time"] = pd.to_datetime(x["time"], utc=True)
    x = (
        x.drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    base = x.set_index("time")

    def resample_bars(rule):
        return (
            base.resample(
                rule,
                label="left",
                closed="left",
            )
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
        )

    # 15 分钟指标
    x15 = resample_bars("15min")
    x15["rsi15"] = rsi(x15["close"], 14)

    ma20 = x15["close"].rolling(20).mean()
    std20 = x15["close"].rolling(20).std(ddof=0)

    x15["bb_lower"] = ma20 - 2 * std20
    x15["bb_upper"] = ma20 + 2 * std20
    x15["prior_low"] = x15["low"].rolling(20).min().shift(1)
    x15["prior_high"] = x15["high"].rolling(20).max().shift(1)

    # 只有到了该 15 分钟 K 线的结束时间，指标才可使用。
    x15 = x15.reset_index()
    x15["time"] = x15["time"] + pd.Timedelta(minutes=15)

    x = pd.merge_asof(
        x.sort_values("time"),
        x15[
            ["time", "rsi15", "bb_lower", "bb_upper",
             "prior_low", "prior_high"]
        ].sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    # 1 小时趋势指标
    x1h = resample_bars("1h")
    x1h["ema50"] = x1h["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    x1h["ema_slope"] = x1h["ema50"].pct_change(3)

    # 1 小时 K 线结束后才能将其指标合并到 5 分钟数据。
    x1h = x1h.reset_index()
    x1h["time"] = x1h["time"] + pd.Timedelta(hours=1)

    x = pd.merge_asof(
        x.sort_values("time"),
        x1h[["time", "ema50", "ema_slope"]].sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    # 5 分钟反转确认
    x["rsi5"] = rsi(x["close"], 14)
    x["recent_low"] = x["low"].rolling(12).min().shift(1)
    x["recent_high"] = x["high"].rolling(12).max().shift(1)

    x["long_confirm"] = (
        (x["close"] > x["recent_high"])
        & (
            x["close"].shift(1)
            <= x["recent_high"].shift(1)
        )
    )

    x["short_confirm"] = (
        (x["close"] < x["recent_low"])
        & (
            x["close"].shift(1)
            >= x["recent_low"].shift(1)
        )
    )

    # 候选区域：仍然使用严格的研究条件，不直接放宽 RSI。
    x["long_candidate"] = (
        (x["rsi15"] < 30)
        & (x["close"] < x["bb_lower"])
        & (x["low"] <= x["prior_low"])
    )

    x["short_candidate"] = (
        (x["rsi15"] > 70)
        & (x["close"] > x["bb_upper"])
        & (x["high"] >= x["prior_high"])
    )

    # 允许候选出现后，在随后 6 根 5 分钟 K 线内确认。
    # rolling(7) 包含当前 K 线及之前 6 根。
    x["long_setup"] = (
        x["long_candidate"]
        .rolling(7, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )

    x["short_setup"] = (
        x["short_candidate"]
        .rolling(7, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )

    x["regime_ok"] = (
        x["ema_slope"].abs() < 0.025
    )

    return x.reset_index(drop=True)


def run_backtest(
    df,
    initial_capital=10000.0,
    risk_pct=0.005,
    fee_bps=5.0,
    slippage_bps=2.0,
    min_rr=2.0,
):
    x = add_indicators(df)

    if initial_capital <= 0:
        raise ValueError("初始资金必须大于 0。")
    if not 0 < risk_pct <= 0.01:
        raise ValueError("单笔风险比例必须大于 0 且不超过 1%。")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("手续费和滑点不能为负数。")
    if min_rr < 1:
        raise ValueError("最低盈亏比不能小于 1。")

    equity = float(initial_capital)
    peak = equity
    max_dd = 0.0
    trades = []
    curve = []
    position = None
    fee_rate = fee_bps / 10000
    slip_rate = slippage_bps / 10000
    cooldown_until = -1

    # 诊断：统计每一层条件的数量。
    diag = {
        "bars": len(x),
        "long_candidate": int(x["long_candidate"].sum()),
        "short_candidate": int(x["short_candidate"].sum()),
        "long_confirm": int(x["long_confirm"].sum()),
        "short_confirm": int(x["short_confirm"].sum()),
        "regime_ok": int(x["regime_ok"].fillna(False).sum()),
        "long_setup_confirm": int(
            (x["long_setup"] & x["long_confirm"]).sum()
        ),
        "short_setup_confirm": int(
            (x["short_setup"] & x["short_confirm"]).sum()
        ),
        "filtered_regime": 0,
        "filtered_risk": 0,
        "entries": 0,
    }

    for i in range(100, len(x) - 1):
        row = x.iloc[i]
        nxt = x.iloc[i + 1]

        curve.append({
            "time": row["time"],
            "equity": equity,
        })

        # 先处理已有仓位的止损止盈。
        if position:
            exit_price = None
            reason = None

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
                exit_exec = exit_price * (
                    1 - slip_rate
                    if position["side"] == "LONG"
                    else 1 + slip_rate
                )

                direction = (
                    1 if position["side"] == "LONG" else -1
                )

                gross = (
                    (exit_exec - position["entry_exec"])
                    * position["qty"] * direction
                )

                fees = (
                    position["entry_exec"] + exit_exec
                ) * position["qty"] * fee_rate

                net = gross - fees
                equity += net

                trades.append({
                    **position,
                    "exit_time": row["time"],
                    "exit_price": exit_price,
                    "reason": reason,
                    "gross_pnl": gross,
                    "fees_est": fees,
                    "net_pnl": net,
                    "equity_after": equity,
                })

                position = None
                cooldown_until = i + 2
                peak = max(peak, equity)
                max_dd = max(
                    max_dd,
                    (peak - equity) / peak if peak else 0,
                )
                continue

        if position or i < cooldown_until:
            continue

        long_signal = (
            bool(row["long_setup"])
            and bool(row["long_confirm"])
        )
        short_signal = (
            bool(row["short_setup"])
            and bool(row["short_confirm"])
        )

        if not long_signal and not short_signal:
            continue

        if not bool(row["regime_ok"]):
            diag["filtered_regime"] += 1
            continue

        # 若同一根 K 线同时触发多空，跳过，避免方向歧义。
        if long_signal and short_signal:
            continue

        side = "LONG" if long_signal else "SHORT"
        entry = float(nxt["open"])

        if side == "LONG":
            stop = float(
                min(row["low"], row["recent_low"]) * (1 - 0.001)
            )
            risk_per_unit = entry - stop
            target = entry + min_rr * risk_per_unit
        else:
            stop = float(
                max(row["high"], row["recent_high"]) * (1 + 0.001)
            )
            risk_per_unit = stop - entry
            target = entry - min_rr * risk_per_unit

        if (
            not np.isfinite(risk_per_unit)
            or risk_per_unit <= 0
            or target <= 0
            or not np.isfinite(entry)
        ):
            diag["filtered_risk"] += 1
            continue

        cost_per_unit = entry * (
            2 * fee_rate + 2 * slip_rate
        )

        if min_rr * risk_per_unit <= cost_per_unit:
            diag["filtered_risk"] += 1
            continue

        risk_cash = equity * risk_pct
        qty = risk_cash / risk_per_unit

        # 原型限制：名义仓位不超过当前权益的 1 倍。
        qty = min(qty, equity / entry)

        if qty <= 0 or not np.isfinite(qty):
            diag["filtered_risk"] += 1
            continue

        entry_exec = entry * (
            1 + slip_rate if side == "LONG" else 1 - slip_rate
        )

        position = {
            "side": side,
            "entry_time": nxt["time"],
            "entry_price": entry,
            "entry_exec": entry_exec,
            "stop": stop,
            "target": target,
            "qty": qty,
            "risk_budget": risk_cash,
            "signal_time": row["time"],
        }

        diag["entries"] += 1

    # 数据结束时按最后收盘价平仓。
    if position and len(x):
        last = x.iloc[-1]
        exit_price = float(last["close"])

        exit_exec = exit_price * (
            1 - slip_rate
            if position["side"] == "LONG"
            else 1 + slip_rate
        )

        direction = 1 if position["side"] == "LONG" else -1

        gross = (
            (exit_exec - position["entry_exec"])
            * position["qty"] * direction
        )

        fees = (
            position["entry_exec"] + exit_exec
        ) * position["qty"] * fee_rate

        net = gross - fees
        equity += net

        trades.append({
            **position,
            "exit_time": last["time"],
            "exit_price": exit_price,
            "reason": "end_of_data",
            "gross_pnl": gross,
            "fees_est": fees,
            "net_pnl": net,
            "equity_after": equity,
        })

        peak = max(peak, equity)
        max_dd = max(
            max_dd,
            (peak - equity) / peak if peak else 0,
        )

    tdf = pd.DataFrame(trades)

    if len(tdf):
        wins = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
        losses = -tdf.loc[tdf["net_pnl"] < 0, "net_pnl"].sum()

        pf = (
            float(wins / losses)
            if losses > 0
            else (None if wins == 0 else float("inf"))
        )

        win_rate = float(
            (tdf["net_pnl"] > 0).mean() * 100
        )
        avg_trade = float(tdf["net_pnl"].mean())
    else:
        pf, win_rate, avg_trade = None, 0.0, 0.0

    metrics = {
        "trades": int(len(tdf)),
        "net_pnl": float(equity - initial_capital),
        "return_pct": float(
            (equity / initial_capital - 1) * 100
        ),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_trade": avg_trade,
        "max_drawdown_pct": float(max_dd * 100),
        "final_equity": float(equity),
    }

    return {
        "metrics": metrics,
        "trades": trades,
        "equity": pd.DataFrame(curve),
        "diagnostics": diag,
    }
