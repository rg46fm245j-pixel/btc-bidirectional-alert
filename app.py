import streamlit as st
import pandas as pd
import numpy as np

from strategy import fetch_klines, run_backtest

st.set_page_config(
    page_title="BTC Reversal Intelligence v1.4",
    layout="wide",
)

st.title("BTC Reversal Intelligence v1.4")
st.caption(
    "Kraken BTC 期货反转策略 · 历史回测与诊断 · 不发送真实订单"
)

with st.sidebar:
    st.header("回测设置")

    days = st.selectbox(
        "历史数据范围（天）",
        [7, 14, 30, 60, 90],
        index=0,
    )

    capital = st.number_input(
        "模拟初始资金（USDT）",
        min_value=100.0,
        value=10000.0,
        step=500.0,
    )

    risk_pct = st.number_input(
        "单笔风险预算（%）",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )

    fee_bps = st.number_input(
        "单边手续费（bps）",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )

    slippage_bps = st.number_input(
        "单边滑点假设（bps）",
        min_value=0.0,
        value=2.0,
        step=0.5,
    )

    min_rr = st.number_input(
        "最低目标盈亏比",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.25,
    )

    max_hold_bars = st.number_input(
        "最长持仓（5分钟K线根数）",
        min_value=1,
        max_value=2880,
        value=288,
        step=12,
    )

st.warning(
    "本程序使用 Kraken Futures 的 PF_XBTUSD 公开行情，并非 Binance BTCUSDT。"
    "这是历史回测原型，不含历史订单簿重放和精确资金费率，不会发送真实订单。"
)

if st.button("获取行情并运行回测", type="primary"):
    try:
        with st.spinner("正在获取行情并运行回测……"):
            df = fetch_klines(
                days=int(days),
                interval="5m",
            )

            if df is None or df.empty:
                st.error("行情接口没有返回可用的 K 线数据。")
                st.stop()

            if "timestamp" not in df.columns:
                if "time" in df.columns:
                    df = df.rename(
                        columns={"time": "timestamp"}
                    )
                else:
                    st.error(
                        "行情数据缺少时间列。实际列名："
                        + ", ".join(map(str, df.columns))
                    )
                    st.stop()

            df = df.copy()
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce",
            )

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )

            df = df.dropna(
                subset=[
                    "timestamp", "open", "high", "low", "close"
                ]
            )
            df = df.drop_duplicates("timestamp")
            df = df.sort_values("timestamp").reset_index(drop=True)

            if df.empty:
                st.error("清理无效行情后没有剩余数据。")
                st.stop()

            first_time = df["timestamp"].min()
            last_time = df["timestamp"].max()
            actual_days = (
                last_time - first_time
            ).total_seconds() / 86400

            st.success(
                f"行情获取成功：共 {len(df)} 根已收盘 K 线"
            )

            c1, c2 = st.columns(2)
            c1.write(f"数据开始时间：{first_time}")
            c2.write(f"数据结束时间：{last_time}")
            c1.write(f"实际覆盖：{actual_days:.2f} 天")
            c2.write(
                f"最新收盘价：{float(df['close'].iloc[-1]):,.2f}"
            )

            if actual_days < days * 0.9:
                st.warning(
                    f"你选择了 {days} 天，但实际覆盖只有 "
                    f"{actual_days:.2f} 天；样本可能不足。"
                )

            result = run_backtest(
                df,
                initial_capital=float(capital),
                risk_pct=float(risk_pct) / 100,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                min_rr=float(min_rr),
                max_hold_bars=int(max_hold_bars),
            )

            summary = result.get("summary", {})

            if not summary:
                st.error(
                    "回测没有返回 summary，请检查 strategy.py。"
                )
                st.stop()

            trades = result.get("trades", pd.DataFrame())
            equity = result.get("equity_curve", pd.DataFrame())
            indicators = result.get("indicators", pd.DataFrame())
            diagnostics = summary.get("diagnostics", {})

            # 回测总览
            st.subheader("回测指标")

            pnl = float(summary.get("total_pnl", 0))
            ret = float(summary.get("total_return_pct", 0))
            pf = summary.get("profit_factor", 0)

            if pf is None:
                pf_text = "N/A"
            elif np.isinf(float(pf)):
                pf_text = "∞"
            else:
                pf_text = f"{float(pf):.2f}"

            cols = st.columns(4)
            cols[0].metric(
                "净收益",
                f"{pnl:.2f} USDT",
                f"{ret:.2f}%",
            )
            cols[1].metric(
                "已结束交易",
                str(int(summary.get("trades", 0))),
            )
            cols[2].metric(
                "胜率",
                f"{float(summary.get('win_rate', 0)):.1f}%",
            )
            cols[3].metric("盈利因子", pf_text)

            cols = st.columns(4)
            cols[0].metric(
                "最大回撤",
                f"{float(summary.get('max_drawdown_pct', 0)):.2f}%",
            )
            cols[1].metric(
                "平均每笔净收益",
                f"{float(summary.get('avg_net_pnl', 0)):.2f} USDT",
            )
            cols[2].metric(
                "期末权益",
                f"{float(summary.get('end_equity', capital)):.2f} USDT",
            )
            cols[3].metric(
                "估算手续费",
                f"{float(summary.get('fees_estimated', 0)):.2f} USDT",
            )

            # 信号诊断
            st.subheader("交易信号诊断")

            diagnostic_labels = {
                "bars": "参与回测的K线数",
                "long_candidate_bars": "做多候选K线数",
                "short_candidate_bars": "做空候选K线数",
                "long_setup_bars": "做多候选有效K线数",
                "short_setup_bars": "做空候选有效K线数",
                "long_confirm_bars": "做多确认K线数",
                "short_confirm_bars": "做空确认K线数",
                "long_setup_and_confirm": "做多候选且确认",
                "short_setup_and_confirm": "做空候选且确认",
                "regime_ok_bars": "环境过滤通过K线数",
                "entries": "实际开仓次数",
                "closed_trades": "已结束交易数",
                "filtered_regime": "被环境过滤拒绝",
                "filtered_risk": "被风险检查拒绝",
                "filtered_invalid_stop": "无效止损拒绝",
                "filtered_gap_at_entry": "开盘跳过止损拒绝",
                "filtered_cooldown": "冷却期过滤",
                "filtered_existing_position": "持仓期间忽略信号",
                "filtered_ambiguous_signal": "多空信号冲突",
                "exit_stop": "止损退出",
                "exit_target": "止盈退出",
                "exit_time": "时间退出",
                "exit_end_of_data": "数据结束退出",
            }

            diag_df = pd.DataFrame([
                {
                    "诊断项目": label,
                    "数值": int(diagnostics.get(key, 0)),
                }
                for key, label in diagnostic_labels.items()
            ])

            st.dataframe(
                diag_df,
                use_container_width=True,
                hide_index=True,
            )

            if int(summary.get("trades", 0)) < 30:
                st.warning(
                    "交易样本少于 30 笔，不能据此判断长期策略优势。"
                    "请根据上面的信号诊断定位瓶颈，不要先放宽策略条件。"
                )

            if pnl < 0:
                st.warning(
                    "本次回测净亏损。请先检查交易明细、成交假设和"
                    "过滤统计，不要为了增加交易次数而随意放宽条件。"
                )

            # 权益曲线
            st.subheader("权益曲线")

            if (
                isinstance(equity, pd.DataFrame)
                and not equity.empty
                and {"timestamp", "equity"}.issubset(equity.columns)
            ):
                curve = equity[["timestamp", "equity"]].copy()
                curve["timestamp"] = pd.to_datetime(
                    curve["timestamp"],
                    utc=True,
                    errors="coerce",
                )
                curve["equity"] = pd.to_numeric(
                    curve["equity"],
                    errors="coerce",
                )
                curve = curve.dropna().sort_values("timestamp")

                if not curve.empty:
                    st.line_chart(
                        curve.set_index("timestamp")["equity"]
                    )

                    peaks = curve["equity"].cummax()
                    dd = (
                        (peaks - curve["equity"])
                        / peaks.replace(0, np.nan)
                        * 100
                    )

                    st.caption(
                        f"按曲线估算的最大回撤：{dd.max():.3f}%"
                    )
            else:
                st.info("没有可绘制的权益曲线。")

            # 交易明细
            st.subheader("交易记录")

            trade_df = (
                trades.copy()
                if isinstance(trades, pd.DataFrame)
                else pd.DataFrame(trades)
            )

            if not trade_df.empty:
                st.dataframe(
                    trade_df,
                    use_container_width=True,
                    hide_index=True,
                )

                if "net_pnl" in trade_df.columns:
                    trade_sum = pd.to_numeric(
                        trade_df["net_pnl"],
                        errors="coerce",
                    ).sum()

                    pnl_diff = trade_sum - pnl

                    st.write(
                        f"交易明细净利润合计：{trade_sum:.4f} USDT"
                    )
                    st.write(
                        f"与回测总净收益的差额：{pnl_diff:.8f} USDT"
                    )

                    if abs(pnl_diff) > 0.01:
                        st.error(
                            "交易明细与总净收益不一致，需要检查结算逻辑。"
                        )
                    else:
                        st.success(
                            "交易明细与总净收益核对一致。"
                        )

                st.download_button(
                    "下载交易记录 CSV",
                    data=trade_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="kraken_btc_futures_trades.csv",
                    mime="text/csv",
                )
            else:
                st.info("本次回测没有已结束交易。")

            # 指标与信号
            st.subheader("策略指标与信号数据")

            if (
                isinstance(indicators, pd.DataFrame)
                and not indicators.empty
            ):
                signal_cols = [
                    c for c in [
                        "timestamp",
                        "close",
                        "rsi15",
                        "rsi5",
                        "long_candidate",
                        "short_candidate",
                        "long_setup",
                        "short_setup",
                        "long_confirm",
                        "short_confirm",
                        "regime_ok",
                        "ema_slope",
                        "bb_lower",
                        "bb_upper",
                    ]
                    if c in indicators.columns
                ]

                st.dataframe(
                    indicators[signal_cols].tail(200),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "下载完整策略指标 CSV",
                    data=indicators.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="btc_strategy_indicators.csv",
                    mime="text/csv",
                )
            else:
                st.info("策略没有返回指标数据。")

            st.caption(
                "数据来源：Kraken Futures PF_XBTUSD。回测为简化模拟，"
                "未精确模拟资金费率、强平、订单簿冲击及所有合约规则。"
                "本程序仅用于研究，不会发送真实订单。"
            )

    except Exception as e:
        st.error(f"运行失败：{type(e).__name__}: {e}")
        st.exception(e)

else:
    st.markdown("""
### 当前版本功能
- 获取 Kraken Futures 公开 K 线
- 15 分钟候选区域与 5 分钟反转确认
- 1 小时环境过滤
- 可调整风险预算、止损止盈、手续费与滑点
- 回测指标、权益曲线、交易明细
- 信号过滤诊断与 CSV 导出

### 尚未包含
- 历史订单簿深度回放
- 精确历史资金费率结算
- 可靠的实时模拟成交系统
- 手机推送告警
- 真实自动下单

先验证历史数据、信号诊断和结算一致性，再考虑下一阶段。
""")
