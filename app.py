import streamlit as st
import pandas as pd
import numpy as np
from strategy import fetch_klines, run_backtest

st.set_page_config(
    page_title="BTC Reversal Intelligence v1.3",
    layout="wide"
)

st.title("BTC Reversal Intelligence v1.3")
st.caption(
    "Kraken BTC 期货反转策略 · 历史回测与诊断 · 不发送真实订单"
)

with st.sidebar:
    st.header("回测设置")

    days = st.selectbox(
        "历史数据范围（天）",
        [7, 14, 30, 60, 90],
        index=0
    )

    capital = st.number_input(
        "模拟初始资金（USDT）",
        min_value=100.0,
        value=10000.0,
        step=500.0
    )

    risk_pct = st.number_input(
        "单笔风险预算（%）",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    fee_bps = st.number_input(
        "单边手续费（bps）",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    slippage_bps = st.number_input(
        "单边滑点假设（bps）",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

    min_rr = st.number_input(
        "最低目标盈亏比",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.25
    )

    max_hold_bars = st.number_input(
        "最长持仓（5分钟K线根数）",
        min_value=1,
        max_value=2880,
        value=288,
        step=12
    )

st.warning(
    "本程序使用 Kraken Futures 的 PF_XBTUSD 公开行情，"
    "并非 Binance BTCUSDT。当前为历史回测原型，"
    "没有历史订单簿重放，也没有精确计入资金费率。"
    "不会发送真实交易订单。"
)

if st.button("获取行情并运行回测", type="primary"):

    try:
        with st.spinner("正在获取行情并运行回测……"):

            # 1. 获取历史行情
            df = fetch_klines(
                days=int(days),
                interval="5m"
            )

            if df is None or df.empty:
                st.error("行情接口没有返回可用的 K 线数据。")
                st.stop()

            # 2. 兼容 timestamp / time 两种时间列名称
            if "timestamp" in df.columns:
                time_col = "timestamp"
            elif "time" in df.columns:
                time_col = "time"
            else:
                st.error(
                    "行情数据缺少时间列。实际列名为："
                    + ", ".join(map(str, df.columns))
                )
                st.stop()

            df = df.copy()
            df[time_col] = pd.to_datetime(
                df[time_col],
                utc=True,
                errors="coerce"
            )

            df = df.dropna(subset=[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)

            required_columns = [
                "open", "high", "low", "close"
            ]
            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing:
                st.error(
                    "行情数据缺少必要字段："
                    + ", ".join(missing)
                )
                st.stop()

            if df.empty:
                st.error("清理无效时间数据后，没有剩余 K 线。")
                st.stop()

            first_time = df[time_col].min()
            last_time = df[time_col].max()

            actual_days = (
                last_time - first_time
            ).total_seconds() / 86400

            st.success(
                f"行情获取成功：共 {len(df)} 根 K 线"
            )

            c1, c2 = st.columns(2)
            c1.write(f"数据开始时间：{first_time}")
            c2.write(f"数据结束时间：{last_time}")

            c1.write(f"实际覆盖：{actual_days:.2f} 天")
            c2.write(
                f"最新收盘价："
                f"{float(df['close'].iloc[-1]):,.2f}"
            )

            if actual_days < days * 0.9:
                st.warning(
                    f"你选择了 {days} 天，"
                    f"但实际覆盖只有 {actual_days:.2f} 天。"
                    "当前回测可能缺少足够的历史样本。"
                )

            # 3. 执行历史回测
            result = run_backtest(
                df,
                initial_capital=float(capital),
                risk_pct=float(risk_pct) / 100,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                min_rr=float(min_rr),
                max_hold_bars=int(max_hold_bars)
            )

            # 当前 strategy.py 返回 summary，而不是 metrics
            summary = result.get("summary", {})

            if not summary:
                st.error(
                    "回测没有返回 summary。"
                    "请检查 strategy.py 的返回结构。"
                )
                st.stop()

            trades = result.get("trades", pd.DataFrame())
            equity = result.get(
                "equity_curve",
                pd.DataFrame()
            )
            indicators = result.get(
                "indicators",
                pd.DataFrame()
            )

            diagnostics = summary.get("diagnostics", {})

            # 4. 回测指标
            st.subheader("回测指标")

            net_pnl = float(summary.get("total_pnl", 0))
            return_pct = float(
                summary.get("total_return_pct", 0)
            )
            trade_count = int(summary.get("trades", 0))
            win_rate = float(summary.get("win_rate", 0))
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
                f"{net_pnl:.2f} USDT",
                f"{return_pct:.2f}%"
            )
            cols[1].metric("已结束交易", str(trade_count))
            cols[2].metric("胜率", f"{win_rate:.1f}%")
            cols[3].metric("盈利因子", pf_text)

            cols = st.columns(4)
            cols[0].metric(
                "最大回撤",
                f"{float(summary.get('max_drawdown_pct', 0)):.2f}%"
            )
            cols[1].metric(
                "平均每笔净收益",
                f"{float(summary.get('avg_net_pnl', 0)):.2f} USDT"
            )
            cols[2].metric(
                "期末权益",
                f"{float(summary.get('end_equity', capital)):.2f} USDT"
            )
            cols[3].metric(
                "估算手续费",
                f"{float(summary.get('fees_estimated', 0)):.2f} USDT"
            )

            # 5. 退出原因与诊断
            st.subheader("退出原因与策略诊断")

            exit_keys = {
                "止损退出": "exit_stop",
                "止盈退出": "exit_target",
                "时间退出": "exit_time",
                "数据结束退出": "exit_end_of_data"
            }

            cols = st.columns(4)
            for col, (label, key) in zip(
                cols, exit_keys.items()
            ):
                col.metric(
                    label,
                    str(diagnostics.get(key, 0))
                )

            if diagnostics.get("exit_end_of_data", 0) > 0:
                st.warning(
                    "存在因数据结束而退出的交易。"
                    "这些交易不一定满足最长持仓时间，"
                    "请在评估策略时单独关注。"
                )

            if trade_count < 30:
                st.warning(
                    "当前交易样本少于 30 笔，"
                    "不足以可靠判断策略是否具有长期优势。"
                )

            if net_pnl < 0:
                st.warning(
                    "本次回测净亏损。"
                    "请先分析交易明细，不要为了提高交易次数"
                    "而随意放宽策略条件。"
                )

            if diagnostics:
                diagnostic_df = pd.DataFrame([
                    {"诊断项目": k, "数量": v}
                    for k, v in diagnostics.items()
                ])
                st.dataframe(
                    diagnostic_df,
                    use_container_width=True,
                    hide_index=True
                )

            # 6. 权益曲线
            st.subheader("权益曲线")

            if (
                isinstance(equity, pd.DataFrame)
                and not equity.empty
                and {"timestamp", "equity"}.issubset(
                    equity.columns
                )
            ):
                curve = equity[
                    ["timestamp", "equity"]
                ].copy()

                curve["timestamp"] = pd.to_datetime(
                    curve["timestamp"],
                    utc=True,
                    errors="coerce"
                )
                curve["equity"] = pd.to_numeric(
                    curve["equity"],
                    errors="coerce"
                )

                curve = curve.dropna(
                    subset=["timestamp", "equity"]
                ).sort_values("timestamp")

                if not curve.empty:
                    st.line_chart(
                        curve.set_index("timestamp")["equity"]
                    )

                    peaks = curve["equity"].cummax()
                    drawdowns = (
                        (peaks - curve["equity"])
                        / peaks.replace(0, np.nan)
                        * 100
                    )

                    st.caption(
                        "按权益曲线估算的最大回撤："
                        f"{drawdowns.max():.3f}%"
                    )
            else:
                st.info("当前没有可绘制的权益曲线。")

            # 7. 交易记录
            st.subheader("交易记录")

            if isinstance(trades, pd.DataFrame):
                trade_df = trades.copy()
            else:
                trade_df = pd.DataFrame(trades)

            if not trade_df.empty:
                st.dataframe(
                    trade_df,
                    use_container_width=True,
                    hide_index=True
                )

                if "net_pnl" in trade_df.columns:
                    trade_sum = pd.to_numeric(
                        trade_df["net_pnl"],
                        errors="coerce"
                    ).sum()

                    pnl_diff = trade_sum - net_pnl

                    st.write(
                        f"交易明细净利润合计：{trade_sum:.4f} USDT"
                    )
                    st.write(
                        f"与回测净收益的差额：{pnl_diff:.6f} USDT"
                    )

                    if abs(pnl_diff) > 0.01:
                        st.error(
                            "交易明细与回测总净收益不一致，"
                            "需要进一步检查权益结算逻辑。"
                        )

                st.download_button(
                    "下载交易记录 CSV",
                    data=trade_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="kraken_btc_futures_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("本次回测没有已结束的交易。")

            # 8. 信号与指标数据
            st.subheader("策略指标数据")

            if (
                isinstance(indicators, pd.DataFrame)
                and not indicators.empty
            ):
                signal_cols = [
                    col for col in [
                        "timestamp", "close",
                        "rsi15", "rsi5",
                        "long_setup", "short_setup",
                        "long_confirm", "short_confirm",
                        "regime_ok"
                    ]
                    if col in indicators.columns
                ]

                if signal_cols:
                    st.dataframe(
                        indicators[signal_cols].tail(200),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "下载策略指标 CSV",
                        data=indicators.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="btc_strategy_indicators.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("指标表中没有可展示的预期字段。")
            else:
                st.info("策略没有返回指标数据。")

            st.caption(
                "数据来源：Kraken Futures PF_XBTUSD。"
                "回测包含手续费和滑点假设，但未精确模拟"
                "资金费率、强平、市场冲击及所有合约规则。"
                "本程序仅用于研究，不会发送真实订单。"
            )

    except Exception as e:
        st.error(f"运行失败：{type(e).__name__}: {e}")
        st.exception(e)

else:
    st.markdown("""
### 当前版本功能
- 获取 Kraken BTC 期货公开 K 线
- 15 分钟候选区域与 5 分钟反转确认
- 1 小时环境过滤
- 可调整风险预算、止损止盈、手续费与滑点
- 回测收益指标、权益曲线和交易记录
- 下载交易记录与策略指标 CSV

### 尚未包含
- 历史订单簿深度回放
- 精确历史资金费率结算
- 可靠的实时模拟成交系统
- 手机推送告警
- 真实自动下单

请先验证历史数据和回测结果，再考虑下一阶段。
""")
