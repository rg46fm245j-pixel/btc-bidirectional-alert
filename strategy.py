import time
import requests
import pandas as pd
import numpy as np

# 历史回测专用：不会发送真实订单
KRAKEN_URL = "https://futures.kraken.com/api/charts/v1/trade/PF_XBTUSD"
INTERVAL_SECONDS = {"5m": 300, "15m": 900, "1h": 3600}


def normalize_timestamp(series):
    """统一为 UTC 时间戳。"""
    return pd.to_datetime(series, utc=True, errors="coerce")


def fetch_klines(days=30, symbol="BTCUSDT", interval="5m"):
    """获取 Kraken Futures PF_XBTUSD K线；symbol 参数仅兼容旧接口。"""
    if interval not in INTERVAL_SECONDS:
        raise ValueError("周期必须是 5m、15m 或 1h")
    if days <= 0:
        raise ValueError("days 必须大于 0")

    seconds = INTERVAL_SECONDS[interval]
    now = int(time.time())
    start = now - int(days * 86400)
    rows = []
    cursor = start
    max_pages = min(500, max(5, int(days * 86400 / seconds / 500) + 8))

    with requests.Session() as session:
        for _ in range(max_pages):
            response = session.get(
                f"{KRAKEN_URL}/{interval}",
                params={"from": cursor, "to": now, "count": 500},
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("candles", [])
            if not batch:
                break

            valid = []
            for candle in batch:
                try:
                    int(candle["time"])
                    valid.append(candle)
                except (KeyError, TypeError, ValueError):
                    continue

            if not valid:
                break

            rows.extend(valid)
            newest = max(int(c["time"]) for c in valid)
            newest_sec = newest // 1000 if newest > 10**11 else newest

            if newest_sec >= now - seconds:
                break

            next_cursor = newest_sec + seconds
            if next_cursor <= cursor:
                break

            cursor = next_cursor
            time.sleep(0.1)

    if not rows:
        raise RuntimeError("Kraken 没有返回 K 线数据，请稍后重试。")

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        raise RuntimeError(f"行情接口返回了意外字段：{list(df.columns)}")

    raw_time = pd.to_numeric(df["time"], errors="coerce")
    valid_times = raw_time.dropna()
    if valid_times.empty:
        raise RuntimeError("行情数据中的时间字段全部无效。")

    unit = "ms" if valid_times.median() > 10**11 else "s"
    df["timestamp"] = pd.to_datetime(
        raw_time, unit=unit, utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise RuntimeError(f"行情数据缺少 {col} 字段。")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["volume"] = pd.to_numeric(
        df["volume"], errors="coerce"
    ).fillna(0.0)

    df = df.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    start_time = pd.Timestamp(start, unit="s", tz="UTC")
    end_time = pd.Timestamp(now, unit="s", tz="UTC")

    df = df[df["timestamp"] >= start_time]
    df = df[
        df["timestamp"] + pd.Timedelta(seconds=seconds) <= end_time
    ]
    df = df.drop_duplicates("timestamp", keep="last")
    df = df.sort_values("timestamp")

    if df.empty:
        raise RuntimeError("没有获取到已收盘的 K 线。")

    return df.reset_index(drop=True)[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]


def rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    avg_loss = losses.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0), 100.0
    )
    result = result.mask(
        (avg_gain == 0) & (avg_loss > 0), 0.0
    )
    return result


def add_indicators(df):
    """计算指标；高周期指标只在对应K线收盘后生效。"""
    if df is None or df.empty:
        raise ValueError("K 线数据为空。")

    x = df.copy()

    if "timestamp" not in x.columns:
        if "time" in x.columns:
            x = x.rename(columns={"time": "timestamp"})
        else:
            raise ValueError("数据必须包含 timestamp 或 time 字段。")

    x["timestamp"] = normalize_timestamp(x["timestamp"])

    for col in ["open", "high", "low", "close"]:
        if col not in x.columns:
            raise ValueError(f"数据缺少 {col} 字段。")
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if "volume" not in x.columns:
        x["volume"] = 0.0

    x["volume"] = pd.to_numeric(
        x["volume"], errors="coerce"
    ).fillna(0.0)

    x = x.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )
    x = x.drop_duplicates("timestamp")
    x = x.sort_values("timestamp").reset_index(drop=True)

    if x.empty:
        raise ValueError("清理无效数据后，没有剩余 K 线。")

    base = x.set_index("timestamp").sort_index()

    def resample_bars(rule):
        out = base.resample(
            rule, label="left", closed="left"
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        return out.dropna(
            subset=["open", "high", "low", "close"]
        )

    # 15分钟指标：15分钟K线结束后才可使用
    q = resample_bars("15min")
    q["rsi15"] = rsi(q["close"], 14)

    mid = q["close"].rolling(20, min_periods=20).mean()
    sd = q["close"].rolling(20, min_periods=20).std(ddof=0)

    q["bb_lower"] = mid - 2 * sd
    q["bb_upper"] = mid + 2 * sd
    q["prior_low"] = (
        q["low"].rolling(20, min_periods=20).min().shift(1)
    )
    q["prior_high"] = (
        q["high"].rolling(20, min_periods=20).max().shift(1)
    )

    q = q.reset_index()
    q["timestamp"] = (
        normalize_timestamp(q["timestamp"])
        + pd.Timedelta(minutes=15)
    )

    right15 = q[
        [
            "timestamp", "rsi15", "bb_lower",
            "bb_upper", "prior_low", "prior_high"
        ]
    ].copy()

    x = pd.merge_asof(
        x.sort_values("timestamp").reset_index(drop=True),
        right15.sort_values("timestamp").reset_index(drop=True),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    # 1小时环境过滤：小时K线收盘后才可使用
    h = resample_bars("1h")
    h["ema50"] = h["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    h["ema_slope"] = h["ema50"].pct_change(3)

    h = h.reset_index()
    h["timestamp"] = (
        normalize_timestamp(h["timestamp"])
        + pd.Timedelta(hours=1)
    )

    right1h = h[
        ["timestamp", "ema50", "ema_slope"]
    ].copy()

    x = pd.merge_asof(
        x.sort_values("timestamp").reset_index(drop=True),
        right1h.sort_values("timestamp").reset_index(drop=True),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    # 5分钟反转确认
    x["rsi5"] = rsi(x["close"], 14)
    x["recent_low"] = (
        x["low"].rolling(12, min_periods=12).min().shift(1)
    )
    x["recent_high"] = (
        x["high"].rolling(12, min_periods=12).max().shift(1)
    )

    x["long_confirm"] = (
        (x["close"] > x["recent_high"])
        & (
            x["close"].shift(1)
            <= x["recent_high"].shift(1)
        )
    ).fillna(False)

    x["short_confirm"] = (
        (x["close"] < x["recent_low"])
        & (
            x["close"].shift(1)
            >= x["recent_low"].shift(1)
        )
    ).fillna(False)

    # 保留原策略阈值，不为增加交易次数而放宽条件
    x["long_candidate"] = (
        (x["rsi15"] < 30)
        & (x["close"] < x["bb_lower"])
        & (x["low"] <= x["prior_low"])
    ).fillna(False)

    x["short_candidate"] = (
        (x["rsi15"] > 70)
        & (x["close"] > x["bb_upper"])
        & (x["high"] >= x["prior_high"])
    ).fillna(False)

    # 候选信号在最近7根5分钟K线内有效
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
    ).fillna(False)

    return x.sort_values("timestamp").reset_index(drop=True)


def run_backtest(
    df,
    initial_capital=10000.0,
    risk_pct=0.005,
    fee_bps=5.0,
    slippage_bps=2.0,
    min_rr=2.0,
    max_hold_bars=288,
):
    """历史回测；只模拟交易，不发送真实订单。"""
    if initial_capital <= 0:
        raise ValueError("初始资金必须大于 0。")
    if not 0 < risk_pct <= 0.01:
        raise ValueError("risk_pct 必须大于 0 且不超过 1%。")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("手续费和滑点不能为负数。")
    if min_rr < 1:
        raise ValueError("min_rr 不能小于 1。")
    if not isinstance(max_hold_bars, int) or max_hold_bars < 1:
        raise ValueError("max_hold_bars 必须是正整数。")

    x = add_indicators(df)
    if len(x) < 101:
        raise ValueError("至少需要 101 根 K 线才能回测。")

    equity = float(initial_capital)
    peak_equity = equity
    max_dd = 0.0

    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0

    trades = []
    curve = []
    position = None
    cooldown_until = -1

    diagnostics = {
        "bars": int(len(x)),
        "long_candidate_bars": int(x["long_candidate"].sum()),
        "short_candidate_bars": int(x["short_candidate"].sum()),
        "long_setup_bars": int(x["long_setup"].sum()),
        "short_setup_bars": int(x["short_setup"].sum()),
        "long_confirm_bars": int(x["long_confirm"].sum()),
        "short_confirm_bars": int(x["short_confirm"].sum()),
        "long_setup_and_confirm": int(
            (x["long_setup"] & x["long_confirm"]).sum()
        ),
        "short_setup_and_confirm": int(
            (x["short_setup"] & x["short_confirm"]).sum()
        ),
        "regime_ok_bars": int(x["regime_ok"].sum()),
        "entries": 0,
        "closed_trades": 0,
        "exit_stop": 0,
        "exit_target": 0,
        "exit_time": 0,
        "exit_end_of_data": 0,
        "filtered_regime": 0,
        "filtered_risk": 0,
        "filtered_cooldown": 0,
        "filtered_existing_position": 0,
        "filtered_ambiguous_signal": 0,
        "filtered_invalid_stop": 0,
        "filtered_gap_at_entry": 0,
    }

    def entry_execution(side, price):
        # 开多买入滑点向上；开空卖出滑点向下
        return price * (
            1 + slip_rate if side == "LONG" else 1 - slip_rate
        )

    def exit_execution(side, price):
        # 平多卖出滑点向下；平空买入滑点向上
        return price * (
            1 - slip_rate if side == "LONG" else 1 + slip_rate
        )

    def mark_equity(mark, pos):
        if pos is None:
            return equity

        ex = exit_execution(pos["side"], float(mark))
        direction = 1 if pos["side"] == "LONG" else -1

        gross = (
            (ex - pos["entry_exec"])
            * pos["qty"]
            * direction
        )
        estimated_exit_fee = (
            (pos["entry_exec"] + ex)
            * pos["qty"]
            * fee_rate
        )

        return (
            equity
            + gross
            - pos["entry_fee"]
            - estimated_exit_fee
        )

    def record_curve(timestamp, mark, pos):
        nonlocal peak_equity, max_dd

        value = float(mark_equity(mark, pos))
        peak_equity = max(peak_equity, value)

        if peak_equity > 0:
            max_dd = max(
                max_dd,
                (peak_equity - value) / peak_equity
            )

        curve.append({
            "timestamp": timestamp,
            "equity": value,
            "realized_equity": float(equity),
        })

    def close_position(pos, timestamp, trigger, reason, exec_price):
        nonlocal equity

        direction = 1 if pos["side"] == "LONG" else -1
        gross = (
            (exec_price - pos["entry_exec"])
            * pos["qty"]
            * direction
        )

        exit_fee = (
            (pos["entry_exec"] + exec_price)
            * pos["qty"]
            * fee_rate
        )
        total_fees = pos["entry_fee"] + exit_fee
        net = gross - total_fees
        equity += net

        trades.append({
            **pos,
            "exit_time": timestamp,
            "exit_price": float(trigger),
            "exit_exec": float(exec_price),
            "reason": reason,
            "gross_pnl": float(gross),
            "entry_fee": float(pos["entry_fee"]),
            "exit_fee": float(exit_fee),
            "fees_est": float(total_fees),
            "net_pnl": float(net),
            "equity_after": float(equity),
        })

        diagnostics["closed_trades"] += 1

        reason_key = {
            "stop": "exit_stop",
            "target": "exit_target",
            "time_exit": "exit_time",
            "end_of_data": "exit_end_of_data",
        }.get(reason)

        if reason_key:
            diagnostics[reason_key] += 1

    def get_exit(pos, bar):
        op = float(bar["open"])
        hi = float(bar["high"])
        lo = float(bar["low"])

        side = pos["side"]
        stop = pos["stop"]
        target = pos["target"]

        # 跳空时按开盘价处理；同一根K线同时触及止损止盈时先止损
        if side == "LONG":
            if op <= stop:
                return op, "stop"
            if op >= target:
                return op, "target"
            hit_stop = lo <= stop
            hit_target = hi >= target
        else:
            if op >= stop:
                return op, "stop"
            if op <= target:
                return op, "target"
            hit_stop = hi >= stop
            hit_target = lo <= target

        if hit_stop:
            return stop, "stop"
        if hit_target:
            return target, "target"

        return None, None

    for i in range(100, len(x)):
        row = x.iloc[i]
        closed_this_bar = False

        long_signal = bool(
            row["long_setup"] and row["long_confirm"]
        )
        short_signal = bool(
            row["short_setup"] and row["short_confirm"]
        )
        has_signal = long_signal or short_signal

        if position is not None:
            trigger, reason = get_exit(position, row)
            held = i - position["entry_bar_index"]

            if trigger is None and held >= max_hold_bars:
                trigger = float(row["close"])
                reason = "time_exit"

            if trigger is not None:
                ex = exit_execution(position["side"], trigger)
                close_position(
                    position,
                    row["timestamp"],
                    trigger,
                    reason,
                    ex,
                )
                position = None
                cooldown_until = i + 2
                closed_this_bar = True

        record_curve(
            row["timestamp"], float(row["close"]), position
        )

        if closed_this_bar or i >= len(x) - 1:
            continue

        if position is not None:
            if has_signal:
                diagnostics["filtered_existing_position"] += 1
            continue

        if i < cooldown_until:
            if has_signal:
                diagnostics["filtered_cooldown"] += 1
            continue

        if not has_signal:
            continue

        if long_signal and short_signal:
            diagnostics["filtered_ambiguous_signal"] += 1
            continue

        if not bool(row["regime_ok"]):
            diagnostics["filtered_regime"] += 1
            continue

        side = "LONG" if long_signal else "SHORT"
        stop = float(
            row["low"] if side == "LONG" else row["high"]
        )

        next_bar = x.iloc[i + 1]
        entry_ref = float(next_bar["open"])

        # 下一根开盘已穿越止损价，放弃该信号
        if (
            (side == "LONG" and entry_ref <= stop)
            or (side == "SHORT" and entry_ref >= stop)
        ):
            diagnostics["filtered_gap_at_entry"] += 1
            continue

        risk_distance = (
            entry_ref - stop
            if side == "LONG"
            else stop - entry_ref
        )

        if (
            not np.isfinite([entry_ref, stop, risk_distance]).all()
            or risk_distance <= 0
        ):
            diagnostics["filtered_invalid_stop"] += 1
            diagnostics["filtered_risk"] += 1
            continue

        target = (
            entry_ref + risk_distance * min_rr
            if side == "LONG"
            else entry_ref - risk_distance * min_rr
        )

        if not np.isfinite(target) or target <= 0:
            diagnostics["filtered_risk"] += 1
            continue

        entry_exec = entry_execution(side, entry_ref)
        stop_exec = exit_execution(side, stop)
        direction = 1 if side == "LONG" else -1

        unit_gross_loss = max(
            0.0,
            (entry_exec - stop_exec) * direction
        )

        entry_fee_per_unit = entry_exec * fee_rate
        exit_fee_per_unit = stop_exec * fee_rate
        unit_risk = (
            unit_gross_loss
            + entry_fee_per_unit
            + exit_fee_per_unit
        )
        risk_budget = equity * risk_pct

        if (
            not np.isfinite([
                entry_exec, stop_exec, unit_risk, risk_budget
            ]).all()
            or unit_risk <= 0
            or risk_budget <= 0
        ):
            diagnostics["filtered_risk"] += 1
            continue

        qty = risk_budget / unit_risk

        if not np.isfinite(qty) or qty <= 0:
            diagnostics["filtered_risk"] += 1
            continue

        entry_fee = entry_exec * qty * fee_rate

        position = {
            "side": side,
            "signal_time": row["timestamp"],
            "entry_time": next_bar["timestamp"],
            "entry_bar_index": i + 1,
            "entry_price": float(entry_ref),
            "entry_exec": float(entry_exec),
            "stop": float(stop),
            "target": float(target),
            "qty": float(qty),
            "risk_budget": float(risk_budget),
            "estimated_unit_risk": float(unit_risk),
            "entry_fee": float(entry_fee),
        }

        diagnostics["entries"] += 1

    # 数据结束仍有持仓时，按最后收盘价强制平仓
    if position is not None:
        last = x.iloc[-1]
        trigger = float(last["close"])
        ex = exit_execution(position["side"], trigger)

        close_position(
            position,
            last["timestamp"],
            trigger,
            "end_of_data",
            ex,
        )
        position = None
        record_curve(last["timestamp"], trigger, None)

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)

    if trades_df.empty:
        win_rate = 0.0
        profit_factor = 0.0
        avg_pnl = 0.0
        total_fees = 0.0
        trade_net_sum = 0.0
    else:
        wins = float(
            trades_df.loc[
                trades_df["net_pnl"] > 0, "net_pnl"
            ].sum()
        )
        losses = float(
            -trades_df.loc[
                trades_df["net_pnl"] < 0, "net_pnl"
            ].sum()
        )

        win_rate = float(
            (trades_df["net_pnl"] > 0).mean() * 100
        )

        profit_factor = (
            wins / losses
            if losses > 0
            else (float("inf") if wins > 0 else 0.0)
        )

        avg_pnl = float(trades_df["net_pnl"].mean())
        total_fees = float(trades_df["fees_est"].sum())
        trade_net_sum = float(trades_df["net_pnl"].sum())

    summary = {
        "initial_capital": float(initial_capital),
        "end_equity": float(equity),
        "total_pnl": float(equity - initial_capital),
        "total_return_pct": float(
            (equity / initial_capital - 1) * 100
        ),
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_dd * 100),
        "avg_net_pnl": avg_pnl,
        "fees_estimated": total_fees,
        "trade_net_pnl_sum": trade_net_sum,
        "reconciliation_diff": float(
            (equity - initial_capital) - trade_net_sum
        ),
        "data_start": str(x["timestamp"].iloc[0]),
        "data_end": str(x["timestamp"].iloc[-1]),
        "data_bars": int(len(x)),
        "data_days": float(
            (
                x["timestamp"].iloc[-1]
                - x["timestamp"].iloc[0]
            ).total_seconds() / 86400
        ),
        "diagnostics": diagnostics,
        "simulation_only": True,
        "model_limitations": [
            "数据来自 Kraken Futures PF_XBTUSD，不是币安 BTCUSDT。",
            "使用K线级别的简化成交模拟，无法还原真实订单簿成交。",
            "同一根K线同时触及止损和止盈时，保守地优先按止损处理。",
            "未模拟资金费率、强平、市场冲击和全部交易所合约规则。",
            "信号确认后按下一根K线开盘入场；如果开盘已穿越止损，则跳过该信号。",
        ],
    }

    return {
        "summary": summary,
        "trades": trades_df,
        "equity_curve": curve_df,
        "indicators": x,
    }
