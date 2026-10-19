import time
import requests
import pandas as pd
import numpy as np

KRAKEN_URL = "https://futures.kraken.com/api/charts/v1/trade/PF_XBTUSD"
UTC_US = "datetime64[us, UTC]"


def normalize_time(series):
    return pd.to_datetime(series, utc=True).astype(UTC_US)


def fetch_klines(days=30, symbol="BTCUSDT", interval="5m"):
    """获取 Kraken Futures 公共 K 线。实际数据为 PF_XBTUSD，并非币安 BTCUSDT。"""
    interval_ms_map = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
    if interval not in interval_ms_map:
        raise ValueError("仅支持 5m、15m 和 1h 周期。")
    if days <= 0:
        raise ValueError("days 必须大于 0。")

    interval_ms = interval_ms_map[interval]
    interval_sec = interval_ms // 1000
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(days * 86_400_000)
    cursor_sec = start_ms // 1000
    end_sec = now_ms // 1000

    rows = []
    previous_max_ms = None
    expected_bars = int(days * 86_400 / interval_sec)
    max_pages = max(10, min(1000, expected_bars // 300 + 10))

    with requests.Session() as session:
        for _ in range(max_pages):
            response = session.get(
                f"{KRAKEN_URL}/{interval}",
                params={"from": cursor_sec, "to": end_sec, "count": 500},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("candles", [])

            if not batch:
                break

            valid = []
            for candle in batch:
                try:
                    t = int(candle["time"])
                    if start_ms <= t < now_ms:
                        valid.append(candle)
                except (KeyError, TypeError, ValueError):
                    continue

            if not valid:
                break

            rows.extend(valid)
            current_max_ms = max(int(c["time"]) for c in valid)

            if current_max_ms >= now_ms - interval_ms:
                break

            if previous_max_ms is not None and current_max_ms <= previous_max_ms:
                break

            previous_max_ms = current_max_ms
            next_cursor = current_max_ms // 1000 + interval_sec

            if next_cursor <= cursor_sec:
                next_cursor = cursor_sec + interval_sec
            if next_cursor >= end_sec:
                break

            cursor_sec = next_cursor

            more = payload.get("more_candles")
            if more is False:
                break
            if more is None and len(valid) < 500:
                break

            time.sleep(0.12)

    if not rows:
        raise RuntimeError("Kraken 没有返回 K 线数据，请稍后重试。")

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        raise RuntimeError("行情数据缺少 time 字段。")

    df["open_time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64")
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0
            else:
                raise RuntimeError(f"行情数据缺少 {col} 字段。")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[
        (df["open_time"] >= start_ms)
        & (df["open_time"] + interval_ms <= now_ms)
    ].copy()

    if df.empty:
        raise RuntimeError("没有获取到已收盘的 K 线数据。")

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).astype(UTC_US)
    df["close_time"] = df["open_time"] + interval_ms - 1

    for col in ["quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]:
        if col not in df.columns:
            df[col] = 0

    return df.reset_index(drop=True)


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (loss.rolling(period).sum() == 0), 50.0)
    result = result.mask((avg_gain == 0) & (loss.rolling(period).sum() > 0), 0.0)
    return result


def add_indicators(df):
    if df is None or df.empty:
        raise ValueError("K 线数据为空。")

    x = df.copy()
    x["time"] = normalize_time(x["time"])
    x = x.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in x.columns:
            if col == "volume":
                x[col] = 0.0
            else:
                raise ValueError(f"K 线数据缺少 {col} 字段。")
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    base = x.set_index("time").sort_index()

    def resample_bars(rule):
        out = base.resample(rule, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        out = out.dropna(subset=["open", "high", "low", "close"])
        out.index = pd.DatetimeIndex(out.index).astype(UTC_US)
        return out

    # 15 分钟指标：在对应 K 线收盘后才可用。
    x15 = resample_bars("15min")
    x15["rsi15"] = rsi(x15["close"], 14)
    ma20 = x15["close"].rolling(20).mean()
    std20 = x15["close"].rolling(20).std(ddof=0)
    x15["bb_lower"] = ma20 - 2 * std20
    x15["bb_upper"] = ma20 + 2 * std20
    x15["prior_low"] = x15["low"].rolling(20).min().shift(1)
    x15["prior_high"] = x15["high"].rolling(20).max().shift(1)
    x15 = x15.reset_index()
    x15["time"] = normalize_time(x15["time"] + pd.Timedelta(minutes=15))

    x = pd.merge_asof(
        x.sort_values("time"),
        x15[["time", "rsi15", "bb_lower", "bb_upper", "prior_low", "prior_high"]].sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    # 1 小时趋势指标：在小时 K 线收盘后才可用。
    x1h = resample_bars("1h")
    x1h["ema50"] = x1h["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    x1h["ema_slope"] = x1h["ema50"].pct_change(3)
    x1h = x1h.reset_index()
    x1h["time"] = normalize_time(x1h["time"] + pd.Timedelta(hours=1))

    x = pd.merge_asof(
        x.sort_values("time"),
        x1h[["time", "ema50", "ema_slope"]].sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    x["rsi5"] = rsi(x["close"], 14)
    x["recent_low"] = x["low"].rolling(12).min().shift(1)
    x["recent_high"] = x["high"].rolling(12).max().shift(1)

    x["long_confirm"] = (
        (x["close"] > x["recent_high"])
        & (x["close"].shift(1) <= x["recent_high"].shift(1))
    )
    x["short_confirm"] = (
        (x["close"] < x["recent_low"])
        & (x["close"].shift(1) >= x["recent_low"].shift(1))
    )

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

    x["long_setup"] = x["long_candidate"].rolling(7, min_periods=1).max().fillna(0).astype(bool)
    x["short_setup"] = x["short_candidate"].rolling(7, min_periods=1).max().fillna(0).astype(bool)
    x["regime_ok"] = x["ema_slope"].abs() < 0.025

    return x.sort_values("time").reset_index(drop=True)


def run_backtest(
    df,
    initial_capital=10000.0,
    risk_pct=0.005,
    fee_bps=5.0,
    slippage_bps=2.0,
    min_rr=2.0,
    max_hold_bars=288,
):
    """历史回测，不执行真实订单。5 分钟 K 线下，288 根约为 24 小时。"""
    if initial_capital <= 0:
        raise ValueError("初始资金必须大于 0。")
    if not 0 < risk_pct <= 0.01:
        raise ValueError("单笔风险比例必须大于 0 且不超过 1%。")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("手续费和滑点不能为负数。")
    if min_rr < 1:
        raise ValueError("最低盈亏比不能小于 1。")
    if not isinstance(max_hold_bars, (int, np.integer)) or max_hold_bars < 1:
        raise ValueError("最长持仓 K 线数必须是大于等于 1 的整数。")

    max_hold_bars = int(max_hold_bars)
    x = add_indicators(df)
    if len(x) < 101:
        raise ValueError(f"有效 K 线只有 {len(x)} 根，至少需要 101 根才能开始回测。")

    equity = float(initial_capital)
    peak_equity = equity
    max_dd = 0.0
    trades, curve, events = [], [], []
    position = None
    cooldown_until = -1
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0

    diag = {
        "bars": len(x),
        "long_candidate": int(x["long_candidate"].sum()),
        "short_candidate": int(x["short_candidate"].sum()),
        "long_confirm": int(x["long_confirm"].sum()),
        "short_confirm": int(x["short_confirm"].sum()),
        "regime_ok": int(x["regime_ok"].fillna(False).sum()),
        "long_setup_confirm": int((x["long_setup"] & x["long_confirm"]).sum()),
        "short_setup_confirm": int((x["short_setup"] & x["short_confirm"]).sum()),
        "filtered_regime": 0,
        "filtered_risk": 0,
        "filtered_existing_position": 0,
        "filtered_cooldown": 0,
        "filtered_both_sides": 0,
        "entries": 0,
        "closed_trades": 0,
        "exit_stop": 0,
        "exit_target": 0,
        "exit_time": 0,
        "exit_end_of_data": 0,
    }

    def mark_to_market(mark_price, pos):
        if pos is None:
            return equity
        exit_est = mark_price * (1 - slip_rate if pos["side"] == "LONG" else 1 + slip_rate)
        direction = 1 if pos["side"] == "LONG" else -1
        gross = (exit_est - pos["entry_exec"]) * pos["qty"] * direction
        fees = (pos["entry_exec"] + exit_est) * pos["qty"] * fee_rate
        return equity + gross - fees

    def record_equity(timestamp, mark_price, pos):
        nonlocal peak_equity, max_dd
        value = mark_to_market(mark_price, pos)
        peak_equity = max(peak_equity, value)
        if peak_equity > 0:
            max_dd = max(max_dd, (peak_equity - value) / peak_equity)
        curve.append({
            "time": timestamp,
            "equity": float(value),
            "realized_equity": float(equity),
        })

    def close_position(pos, timestamp, exit_price, reason):
        nonlocal equity
        exit_exec = exit_price * (1 - slip_rate if pos["side"] == "LONG" else 1 + slip_rate)
        direction = 1 if pos["side"] == "LONG" else -1
        gross = (exit_exec - pos["entry_exec"]) * pos["qty"] * direction
        fees = (pos["entry_exec"] + exit_exec) * pos["qty"] * fee_rate
        net = gross - fees
        equity += net

        trades.append({
            **pos,
            "exit_time": timestamp,
            "exit_price": float(exit_price),
            "exit_exec": float(exit_exec),
            "reason": reason,
            "gross_pnl": float(gross),
            "fees_est": float(fees),
            "net_pnl": float(net),
            "equity_after": float(equity),
        })

        events.append({
            "time": timestamp,
            "event": "EXIT",
            "side": pos["side"],
            "price": float(exit_price),
            "reason": reason,
            "detail": f"net_pnl={net:.4f}; fees={fees:.4f}",
        })

        key = {
            "stop": "exit_stop",
            "target": "exit_target",
            "time_exit": "exit_time",
            "end_of_data": "exit_end_of_data",
        }.get(reason)
        if key:
            diag[key] += 1

    for i in range(100, len(x)):
        row = x.iloc[i]
        just_closed = False

        if position is not None:
            exit_price, reason = None, None

            # 同一根 K 线触及止盈和止损时，保守地先按止损处理。
            if position["side"] == "LONG":
                if float(row["low"]) <= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif float(row["high"]) >= position["target"]:
                    exit_price, reason = position["target"], "target"
            else:
                if float(row["high"]) >= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif float(row["low"]) <= position["target"]:
                    exit_price, reason = position["target"], "target"

            held = i - position["entry_bar_index"]
            if exit_price is None and held >= max_hold_bars:
                exit_price, reason = float(row["close"]), "time_exit"

            if exit_price is not None:
                close_position(position, row["time"], float(exit_price), reason)
                position = None
                cooldown_until = i + 2
                just_closed = True

        record_equity(row["time"], float(row["close"]), position)

        # 最后一根 K 线没有下一根 K 线可供开仓。
        if i >= len(x) - 1 or just_closed:
            continue

        long_signal = bool(row["long_setup"] and row["long_confirm"])
        short_signal = bool(row["short_setup"] and row["short_confirm"])
        if not long_signal and not short_signal:
            continue

        side_label = "LONG" if long_signal and not short_signal else (
            "SHORT" if short_signal and not long_signal else "BOTH"
        )
        event = {
            "time": row["time"],
            "event": "SIGNAL",
            "side": side_label,
            "price": float(row["close"]),
            "reason": "",
            "detail": "",
        }

        if long_signal and short_signal:
            diag["filtered_both_sides"] += 1
            event["reason"] = "同时出现多空信号"
            events.append(event)
            continue
        if position is not None:
            diag["filtered_existing_position"] += 1
            event["reason"] = "已有持仓"
            events.append(event)
            continue
        if i < cooldown_until:
            diag["filtered_cooldown"] += 1
            event["reason"] = "平仓冷却期"
            events.append(event)
            continue
        if not bool(row["regime_ok"]):
            diag["filtered_regime"] += 1
            event["reason"] = "趋势过滤未通过"
            events.append(event)
            continue

        side = side_label
        entry = float(x.iloc[i + 1]["open"])

        if side == "LONG":
            stop_ref = min(float(row["low"]), float(row["recent_low"]))
            stop = stop_ref * (1 - 0.001)
            risk_per_unit = entry - stop
            target = entry + min_rr * risk_per_unit
        else:
            stop_ref = max(float(row["high"]), float(row["recent_high"]))
            stop = stop_ref * (1 + 0.001)
            risk_per_unit = stop - entry
            target = entry - min_rr * risk_per_unit

        if (
            not np.isfinite(entry)
            or not np.isfinite(stop)
            or not np.isfinite(target)
            or not np.isfinite(risk_per_unit)
            or risk_per_unit <= 0
            or target <= 0
        ):
            diag["filtered_risk"] += 1
            event["reason"] = "入场价或止损距离无效"
            events.append(event)
            continue

        risk_cash = equity * risk_pct
        qty = min(risk_cash / risk_per_unit, equity / entry)
        if not np.isfinite(qty) or qty <= 0:
            diag["filtered_risk"] += 1
            event["reason"] = "仓位数量无效"
            events.append(event)
            continue

        entry_exec = entry * (1 + slip_rate if side == "LONG" else 1 - slip_rate)
        position = {
            "side": side,
            "entry_time": x.iloc[i + 1]["time"],
            "entry_price": entry,
            "entry_exec": entry_exec,
            "stop": float(stop),
            "target": float(target),
            "qty": float(qty),
            "risk_budget": float(risk_cash),
            "signal_time": row["time"],
            "entry_bar_index": i + 1,
        }
        diag["entries"] += 1
        event["reason"] = "已安排下一根 K 线开仓"
        event["detail"] = (
            f"entry={entry:.4f}; stop={stop:.4f}; target={target:.4f}; "
            f"qty={qty:.8f}; risk_budget={risk_cash:.4f}"
        )
        events.append(event)

    # 最后一根数据到达时仍未平仓，则结算剩余仓位。
    # 若已达到持仓期限，退出原因记为 time_exit。
    if position is not None and len(x):
        last_index = len(x) - 1
        last = x.iloc[last_index]
        held = last_index - position["entry_bar_index"]
        reason = "time_exit" if held >= max_hold_bars else "end_of_data"
        close_position(position, last["time"], float(last["close"]), reason)
        position = None
        record_equity(last["time"], float(last["close"]), None)

    if len(x) and (not curve or curve[-1]["time"] != x.iloc[-1]["time"]):
        last = x.iloc[-1]
        record_equity(last["time"], float(last["close"]), position)

    diag["closed_trades"] = len(trades)
    trade_df = pd.DataFrame(trades)

    if not trade_df.empty:
        wins = float(trade_df.loc[trade_df["net_pnl"] > 0, "net_pnl"].sum())
        losses = float(-trade_df.loc[trade_df["net_pnl"] < 0, "net_pnl"].sum())
        profit_factor = wins / losses if losses > 0 else (float("inf") if wins > 0 else None)
        win_rate = float((trade_df["net_pnl"] > 0).mean() * 100)
        avg_trade = float(trade_df["net_pnl"].mean())
    else:
        profit_factor = None
        win_rate = 0.0
        avg_trade = 0.0

    metrics = {
        "trades": int(len(trade_df)),
        "net_pnl": float(equity - initial_capital),
        "return_pct": float((equity / initial_capital - 1) * 100),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "max_drawdown_pct": float(max_dd * 100),
        "final_equity": float(equity),
    }

    return {
        "metrics": metrics,
        "trades": trades,
        "equity": pd.DataFrame(curve),
        "diagnostics": diag,
        "events": pd.DataFrame(events),
    }
