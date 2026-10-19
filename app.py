import streamlit as st
import pandas as pd
import numpy as np
from strategy import fetch_klines, run_backtest

st.set_page_config(
    page_title="BTC Reversal Intelligence v1.2",
    layout="wide"
)

st.title("BTC Reversal Intelligence v1.2")
st.caption("Kraken BTC 期货反转策略 · 历史回测与逐信号诊断 · 不发送真实订单")

with st.sidebar:
    st.header("回测设置")
    days = st.selectbox(
        "历史数据范围（天）",
        [7, 14, 30, 60, 90],
        index=0
    )
    capital = st.number_input(
        "模拟初始资金 (USDT)",
        min_value=100.0,
        value=10000.0,
        step=500.0
    )
    risk_pct = st.number_input(
        "单笔风险预算 (%)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    fee_bps = st.number_input(
        "单边手续费 (bps)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )
    slippage_bps = st.number_input(
        "单边滑点假设 (bps)",
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
    "原型限制：使用 Kraken PF_XBTUSD 公开 K 线；"
    "没有历史订单簿重放，资金费率暂未纳入。"
    "回测结果不代表未来表现，也不会执行真实交易。"
)

if st.button("获取行情并运行回测", type="primary"):
    with st.spinner("正在获取行情并运行回测……"):
        try:
            df = fetch_klines(days=days, interval="5m")

            if df.empty:
                st.error("行情接口没有返回可用的 K 线数据。")
                st.stop()

            first_time = pd.to_datetime(df["time"], utc=True).min()
            last_time = pd.to_datetime(df["time"], utc=True).max()
            actual_days = (
                last_time - first_time
            ).total_seconds() / 86400

            st.success(f"行情获取成功：共 {len(df)} 根 K 线")
            st.write(f"数据开始时间：{first_time}")
            st.write(f"数据结束时间：{last_time}")
            st.write(f"实际覆盖：{actual_days:.2f} 天")
            st.write(f"最新收盘价：{float(df['close'].iloc[-1]):,.2f}")

            if actual_days < days * 0.9:
                st.warning(
                    f"你选择了 {days} 天，但实际数据只覆盖 "
                    f"{actual_days:.2f} 天。请检查 strategy.py 的分页逻辑。"
                )

            result = run_backtest(
                df,
                initial_capital=capital,
                risk_pct=risk_pct / 100,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                min_rr=min_rr,
                max_hold_bars=int(max_hold_bars),
            )

            metrics = result["metrics"]
            trades = result["trades"]
            diagnostics = result.get("diagnostics", {})
            events = result.get("events", pd.DataFrame())
            equity = result.get("equity", pd.DataFrame())

            st.subheader("回测指标")
            cols = st.columns(4)
            cols[0].metric(
                "净收益",
                f'{metrics["net_pnl"]:.2f} USDT',
                f'{metrics["return_pct"]:.2f}%'
            )
            cols[1].metric("交易笔数", str(metrics["trades"]))
            cols[2].metric("胜率", f'{metrics["win_rate"]:.1f}%')

            pf = metrics["profit_factor"]
            pf_text = (
                "N/A" if pf is None
                else ("∞" if np.isinf(pf) else f"{pf:.2f}")
            )
            cols[3].metric("盈利因子", pf_text)

            cols = st.columns(3)
            cols[0].metric(
                "最大回撤",
                f'{metrics["max_drawdown_pct"]:.2f}%'
            )
            cols[1].metric(
                "平均每笔净收益",
                f'{metrics["avg_trade"]:.2f} USDT'
            )
            cols[2].metric(
                "期末权益",
                f'{metrics["final_equity"]:.2f} USDT'
            )

            st.subheader("退出原因核对")
            exit_keys = {
                "止损退出": "exit_stop",
                "止盈退出": "exit_target",
                "时间退出": "exit_time",
                "数据结束退出": "exit_end_of_data",
            }
            cols = st.columns(4)
            for col, (label, key) in zip(cols, exit_keys.items()):
                col.metric(label, str(diagnostics.get(key, 0)))

            if diagnostics.get("exit_end_of_data", 0) > 0:
                st.warning(
                    "仍有交易因历史数据结束而退出。"
                    "请检查这些交易是否在数据结束前已达到最长持仓时间。"
                )

            if metrics["trades"] < 30:
                st.warning(
                    "交易样本少于 30 笔，暂时不能可靠判断长期盈利能力。"
                )

            if metrics["net_pnl"] < 0:
                st.warning(
                    "本次回测亏损。请先分析信号和交易记录，"
                    "不要仅为增加交易次数而放宽条件。"
                )

            st.subheader("权益曲线与回撤检查")
            if not equity.empty and {"time", "equity"}.issubset(equity.columns):
                curve = equity[["time", "equity"]].copy()
                curve["equity"] = pd.to_numeric(
                    curve["equity"], errors="coerce"
                )
                curve = curve.dropna(
                    subset=["equity"]
                ).sort_values("time")

                if not curve.empty:
                    st.line_chart(curve.set_index("time")["equity"])
                    peaks = curve["equity"].cummax()
                    drawdowns = (
                        (peaks - curve["equity"])
                        / peaks.replace(0, np.nan)
                        * 100
                    )
                    st.caption(
                        "权益曲线计算的最大回撤："
                        f"{drawdowns.max():.3f}%"
                    )

            st.subheader("交易记录")
            if trades:
                trade_df = pd.DataFrame(trades)
                st.dataframe(
                    trade_df,
                    use_container_width=True,
                    hide_index=True
                )

                if {"net_pnl", "reason"}.issubset(trade_df.columns):
                    trade_sum = pd.to_numeric(
                        trade_df["net_pnl"], errors="coerce"
                    ).sum()
                    pnl_diff = trade_sum - metrics["net_pnl"]

                    st.write(f"交易记录净利润合计：{trade_sum:.4f} USDT")
                    st.write(f"与回测总净收益的差额：{pnl_diff:.6f} USDT")

                    if abs(pnl_diff) > 0.01:
                        st.error(
                            "交易明细合计与回测总净收益不一致，"
                            "请检查 strategy.py 的权益结算逻辑。"
                        )

                st.download_button(
                    "下载交易记录 CSV",
                    trade_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="kraken_btc_futures_trades.csv",
                    mime="text/csv",
                )
            else:
                st.info("当前没有已结束的交易记录。")

            st.subheader("策略信号诊断")
            if diagnostics:
                st.dataframe(
                    pd.DataFrame([
                        {"诊断项目": key, "数量": value}
                        for key, value in diagnostics.items()
                    ]),
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader("逐信号事件记录")
            if isinstance(events, pd.DataFrame) and not events.empty:
                event_df = events.sort_values(
                    "time", ascending=False
                )
                st.dataframe(
                    event_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "下载逐信号事件 CSV",
                    event_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="btc_signal_events.csv",
                    mime="text/csv",
                )
            else:
                st.info("当前没有逐信号事件记录。")

            st.caption(
                "数据来源：Kraken PF_XBTUSD，不是 Binance BTCUSDT。"
                "未纳入历史资金费率和历史订单簿重放。"
                "本程序仅用于研究，不会发送真实订单。"
            )

        except Exception as e:
            st.error(f"运行失败：{type(e).__name__}: {e}")
            st.exception(e)
else:
    st.markdown("""
### 当前版本功能
- Kraken BTC 期货公开 K 线
- 1 小时趋势过滤、15 分钟候选区域、5 分钟反转确认
- 风险预算、止损止盈、手续费和滑点假设
- 可调整最长持仓时间
- 收益指标、权益曲线、交易记录与 CSV 下载
- 逐信号事件记录及退出原因诊断

### 尚未包含
- 历史订单簿深度回放
- 精确历史资金费率结算
- 实时模拟成交、告警和真实下单

本原型仅用于研究，不会连接真实下单接口。
""")
