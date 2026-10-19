import streamlit as st
import pandas as pd
import numpy as np
from strategy import fetch_klines, run_backtest

st.set_page_config(
    page_title="BTC Reversal Intelligence v1.1",
    layout="wide"
)

st.title("BTC Reversal Intelligence v1.1")
st.caption("Kraken BTC 期货反转策略 · 历史回测与逐信号诊断 · 不发送真实订单")

with st.sidebar:
    st.header("回测设置")
    days = st.selectbox("历史数据范围（天）", [7, 14, 30, 60, 90], index=0)
    capital = st.number_input("模拟初始资金 (USDT)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("单笔风险预算 (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    fee_bps = st.number_input("单边手续费 (bps)", min_value=0.0, value=5.0, step=0.5)
    slippage_bps = st.number_input("单边滑点假设 (bps)", min_value=0.0, value=2.0, step=0.5)
    min_rr = st.number_input("最低目标盈亏比", min_value=1.0, max_value=5.0, value=2.0, step=0.25)

st.warning(
    "原型限制：使用 Kraken PF_XBTUSD 公开 K 线；"
    "没有历史订单簿重放，资金费率暂未纳入。"
    "回测结果不代表未来表现，也不会执行真实交易。"
)

if st.button("获取行情并运行回测", type="primary"):
    with st.spinner("正在获取行情并运行回测……"):
        try:
            df = fetch_klines(days=days)

            if df.empty:
                st.error("行情接口没有返回可用的 K 线数据。")
                st.stop()

            st.success(f"行情获取成功：共 {len(df)} 根 K 线")
            st.write(f"数据时间范围：{df['time'].min()} 至 {df['time'].max()}")
            st.write(f"最新收盘价：{float(df['close'].iloc[-1]):,.2f}")

            result = run_backtest(
                df,
                initial_capital=capital,
                risk_pct=risk_pct / 100,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                min_rr=min_rr,
            )

            metrics = result["metrics"]
            trades = result["trades"]
            diagnostics = result.get("diagnostics", {})
            events = result.get("events", pd.DataFrame())
            equity = result.get("equity", pd.DataFrame())

            st.subheader("回测指标")
            cols = st.columns(4)
            cols[0].metric("净收益", f'{metrics["net_pnl"]:.2f} USDT', f'{metrics["return_pct"]:.2f}%')
            cols[1].metric("交易笔数", str(metrics["trades"]))
            cols[2].metric("胜率", f'{metrics["win_rate"]:.1f}%')

            pf = metrics["profit_factor"]
            pf_text = "N/A" if pf is None else ("∞" if np.isinf(pf) else f"{pf:.2f}")
            cols[3].metric("盈利因子", pf_text)

            cols = st.columns(3)
            cols[0].metric("最大回撤", f'{metrics["max_drawdown_pct"]:.2f}%')
            cols[1].metric("平均每笔净收益", f'{metrics["avg_trade"]:.2f} USDT')
            cols[2].metric("期末权益", f'{metrics["final_equity"]:.2f} USDT')

            if metrics["trades"] < 30:
                st.warning("交易样本少于 30 笔，暂时不能可靠判断长期盈利能力。")
            if metrics["net_pnl"] < 0:
                st.warning("本次回测亏损。请先分析信号和交易记录，不要仅为增加交易次数而放宽条件。")

            # 权益曲线与回撤交叉检查
            st.subheader("权益曲线与回撤检查")
            if not equity.empty and {"time", "equity"}.issubset(equity.columns):
                curve = equity[["time", "equity"]].copy()
                curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")
                curve = curve.dropna(subset=["equity"]).sort_values("time")

                if not curve.empty:
                    st.line_chart(curve.set_index("time")["equity"])
                    peak_series = curve["equity"].cummax()
                    drawdown_series = (peak_series - curve["equity"]) / peak_series.replace(0, np.nan) * 100
                    curve["回撤百分比"] = drawdown_series
                    st.caption(f"基于当前返回的权益曲线计算的最大回撤：{drawdown_series.max():.3f}%")

                    if metrics["net_pnl"] < 0 and metrics["max_drawdown_pct"] == 0:
                        st.error(
                            "发现统计不一致的可能：策略指标最大回撤为 0，"
                            "但最终净收益为负。当前曲线还可能没有包含完整的逐笔权益变化；"
                            "需在 strategy.py 中修正权益曲线和回撤计算。"
                        )
            else:
                st.info("当前没有可用于绘图的权益曲线。")

            # 信号诊断
            st.subheader("策略信号诊断")
            st.caption("信号、过滤和交易数量统计口径不同，不能简单相减。")

            if diagnostics:
                labels = {
                    "bars": "参与分析的 K 线",
                    "long_candidate": "多头候选",
                    "short_candidate": "空头候选",
                    "long_confirm": "多头确认",
                    "short_confirm": "空头确认",
                    "regime_ok": "通过趋势过滤的 K 线",
                    "long_setup_confirm": "多头组合信号",
                    "short_setup_confirm": "空头组合信号",
                    "filtered_regime": "趋势过滤次数",
                    "filtered_risk": "风险条件过滤次数",
                    "filtered_existing_position": "已有持仓而跳过",
                    "filtered_cooldown": "冷却期跳过",
                    "filtered_both_sides": "多空同时触发而跳过",
                    "entries": "实际开仓次数",
                    "closed_trades": "已结束交易数",
                }

                preferred = list(labels.keys())
                cols = st.columns(4)
                for idx, key in enumerate(preferred):
                    if key in diagnostics:
                        cols[idx % 4].metric(labels[key], str(diagnostics[key]))

                diag_df = pd.DataFrame([
                    {"诊断项目": labels.get(k, k), "数量": v}
                    for k, v in diagnostics.items()
                ])
                with st.expander("查看完整诊断表", expanded=True):
                    st.dataframe(diag_df, use_container_width=True, hide_index=True)

            else:
                st.info("没有收到 diagnostics。请确认线上 strategy.py 已更新。")

            # 新增：逐信号事件记录
            st.subheader("逐信号事件记录")
            if isinstance(events, pd.DataFrame) and not events.empty:
                event_df = events.copy()
                if "time" in event_df.columns:
                    event_df = event_df.sort_values("time", ascending=False)

                st.caption("查看每次组合信号的方向、价格、处理结果与跳过原因。")
                st.dataframe(event_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "下载逐信号事件 CSV",
                    event_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="btc_signal_events.csv",
                    mime="text/csv",
                )

                if "reason" in event_df.columns:
                    skipped = event_df[
                        event_df["reason"].astype(str).str.contains(
                            "跳过|过滤|无效|不足|已有持仓|冷却期|同时出现",
                            regex=True,
                        )
                    ]
                    if not skipped.empty:
                        with st.expander("只看被跳过或被过滤的事件"):
                            st.dataframe(skipped, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "没有逐信号事件表。请确认 strategy.py 的 run_backtest "
                    "返回了 events；若仍没有，可能是线上代码尚未更新。"
                )

            # 交易记录
            st.subheader("交易记录")
            if trades:
                trade_df = pd.DataFrame(trades)
                st.dataframe(trade_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "下载交易记录 CSV",
                    trade_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="kraken_btc_futures_trades.csv",
                    mime="text/csv",
                )

                if "net_pnl" in trade_df.columns:
                    win_count = int((trade_df["net_pnl"] > 0).sum())
                    loss_count = int((trade_df["net_pnl"] < 0).sum())
                    flat_count = int((trade_df["net_pnl"] == 0).sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("盈利交易", win_count)
                    c2.metric("亏损交易", loss_count)
                    c3.metric("盈亏持平", flat_count)

                # 成本分解
                if {"gross_pnl", "fees_est", "net_pnl"}.issubset(trade_df.columns):
                    st.subheader("交易成本分析")
                    gross_total = pd.to_numeric(trade_df["gross_pnl"], errors="coerce").sum()
                    fee_total = pd.to_numeric(trade_df["fees_est"], errors="coerce").sum()
                    net_total = pd.to_numeric(trade_df["net_pnl"], errors="coerce").sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("累计毛盈亏", f"{gross_total:.2f} USDT")
                    c2.metric("累计手续费估算", f"{fee_total:.2f} USDT")
                    c3.metric("累计净盈亏", f"{net_total:.2f} USDT")

                if "reason" in trade_df.columns:
                    st.caption("退出原因说明：stop=止损，target=止盈，end_of_data=数据结束时按收盘价平仓。")
            else:
                st.info("当前没有已结束的交易记录。请先检查信号事件表，确认信号是否出现及其处理原因。")

            st.caption(
                "注意：当前使用 Kraken PF_XBTUSD，不是 Binance BTCUSDT。"
                "回测为研究原型，未计入历史资金费率，不能代表未来表现。"
            )

        except Exception as e:
            st.error(f"运行失败：{type(e).__name__}: {e}")
            st.info("请把完整错误信息发给助手，以便检查。")

else:
    st.markdown("""
### 当前版本功能
- Kraken BTC 期货公开 K 线
- 1 小时趋势过滤、15 分钟候选区域、5 分钟反转确认
- 风险预算、止损止盈、手续费和滑点假设
- 收益指标、权益曲线、交易记录与 CSV 下载
- 逐信号事件记录及跳过原因（需要 strategy.py 返回 events）

### 尚未包含
- 历史订单簿深度回放
- 精确历史资金费率结算
- 实时模拟成交、告警和真实下单

本原型仅用于研究，不会连接真实下单接口。
""")
