import streamlit as st
import pandas as pd
import numpy as np
from strategy import fetch_klines, run_backtest

st.set_page_config(
    page_title="BTC Reversal Intelligence v1",
    layout="wide"
)

st.title("BTC Reversal Intelligence v1")
st.caption(
    "Kraken BTC 期货反转策略 · 历史回测原型 · 不会发送真实订单"
)

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

st.warning(
    "原型限制：使用 Kraken 公开 BTC 期货 K 线；"
    "不包含历史订单簿重放；资金费率暂未纳入；"
    "回测结果不代表未来表现。"
)

if st.button("获取行情并运行回测", type="primary"):

    with st.spinner("正在获取行情并运行回测……"):

        try:
            df = fetch_klines(days=days)

            if df.empty:
                st.error("行情接口没有返回可用的 K 线数据。")
                st.stop()

            st.success(f"行情获取成功：共 {len(df)} 根 K 线")

            st.write(
                f"数据时间范围：{df['time'].min()} 至 {df['time'].max()}"
            )

            st.write(
                f"最新收盘价：{float(df['close'].iloc[-1]):,.2f}"
            )

            result = run_backtest(
                df,
                initial_capital=capital,
                risk_pct=risk_pct / 100,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                min_rr=min_rr
            )

            trades = result["trades"]
            metrics = result["metrics"]
            diagnostics = result.get("diagnostics", {})

            # -------------------------
            # 回测指标
            # -------------------------
            st.subheader("回测指标")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "净收益",
                f'{metrics["net_pnl"]:.2f} USDT',
                f'{metrics["return_pct"]:.2f}%'
            )

            c2.metric(
                "交易笔数",
                str(metrics["trades"])
            )

            c3.metric(
                "胜率",
                f'{metrics["win_rate"]:.1f}%'
            )

            pf = metrics["profit_factor"]

            if pf is None:
                pf_text = "N/A"
            elif np.isinf(pf):
                pf_text = "∞"
            else:
                pf_text = f"{pf:.2f}"

            c4.metric("盈利因子", pf_text)

            c5, c6, c7 = st.columns(3)

            c5.metric(
                "最大回撤",
                f'{metrics["max_drawdown_pct"]:.2f}%'
            )

            c6.metric(
                "平均每笔净收益",
                f'{metrics["avg_trade"]:.2f} USDT'
            )

            c7.metric(
                "期末权益",
                f'{metrics["final_equity"]:.2f} USDT'
            )

            if metrics["trades"] < 30:
                st.warning(
                    "当前交易样本少于 30 笔，"
                    "暂时不能可靠判断策略的长期盈利能力。"
                )

            if metrics["net_pnl"] < 0:
                st.info(
                    "本次回测为亏损结果。"
                    "请先查看信号诊断和交易记录，"
                    "不要仅通过放宽条件来提高交易次数。"
                )

            # -------------------------
            # 策略信号诊断
            # -------------------------
            st.subheader("策略信号诊断")
            st.caption(
                "这些数据用于定位信号在哪一层减少。"
                "不同指标的统计口径不同，不能简单相减。"
            )

            if diagnostics:
                d1, d2, d3, d4 = st.columns(4)

                d1.metric(
                    "参与分析的 K 线",
                    str(diagnostics.get("bars", 0))
                )

                d2.metric(
                    "多头候选",
                    str(diagnostics.get("long_candidate", 0))
                )

                d3.metric(
                    "空头候选",
                    str(diagnostics.get("short_candidate", 0))
                )

                d4.metric(
                    "通过趋势过滤的 K 线",
                    str(diagnostics.get("regime_ok", 0))
                )

                d5, d6, d7, d8 = st.columns(4)

                d5.metric(
                    "多头确认",
                    str(diagnostics.get("long_confirm", 0))
                )

                d6.metric(
                    "空头确认",
                    str(diagnostics.get("short_confirm", 0))
                )

                d7.metric(
                    "多头组合信号",
                    str(diagnostics.get("long_setup_confirm", 0))
                )

                d8.metric(
                    "空头组合信号",
                    str(diagnostics.get("short_setup_confirm", 0))
                )

                d9, d10, d11 = st.columns(3)

                d9.metric(
                    "趋势条件过滤次数",
                    str(diagnostics.get("filtered_regime", 0))
                )

                d10.metric(
                    "风险条件过滤次数",
                    str(diagnostics.get("filtered_risk", 0))
                )

                d11.metric(
                    "实际开仓次数",
                    str(diagnostics.get("entries", 0))
                )

                diagnostic_labels = {
                    "bars": "参与分析的 K 线",
                    "long_candidate": "多头候选信号",
                    "short_candidate": "空头候选信号",
                    "long_confirm": "多头确认信号",
                    "short_confirm": "空头确认信号",
                    "regime_ok": "通过趋势过滤的 K 线",
                    "long_setup_confirm": "多头组合信号",
                    "short_setup_confirm": "空头组合信号",
                    "filtered_regime": "趋势过滤次数",
                    "filtered_risk": "风险条件过滤次数",
                    "entries": "实际开仓次数",
                }

                diagnostic_df = pd.DataFrame([
                    {
                        "诊断项目": diagnostic_labels.get(key, key),
                        "数量": value,
                    }
                    for key, value in diagnostics.items()
                ])

                with st.expander("查看完整诊断表"):
                    st.dataframe(
                        diagnostic_df,
                        use_container_width=True,
                        hide_index=True
                    )

                candidate_total = (
                    diagnostics.get("long_candidate", 0)
                    + diagnostics.get("short_candidate", 0)
                )

                combined_total = (
                    diagnostics.get("long_setup_confirm", 0)
                    + diagnostics.get("short_setup_confirm", 0)
                )

                if candidate_total == 0:
                    st.warning(
                        "没有出现多头或空头候选信号。"
                        "应优先检查 RSI、布林带和前高前低条件，"
                        "而不是先修改止损止盈。"
                    )
                elif combined_total == 0:
                    st.warning(
                        "出现了候选信号，但没有形成组合确认信号。"
                        "请结合候选数量、确认数量和趋势过滤数量定位瓶颈。"
                    )
                elif diagnostics.get("entries", 0) == 0:
                    st.warning(
                        "已经形成组合信号，但没有实际开仓。"
                        "请检查趋势过滤、风险过滤和开仓条件。"
                    )
                else:
                    st.success(
                        "本次已有实际开仓。"
                        "下一步应检查每笔交易的进出场、交易成本和盈亏。"
                    )

            else:
                st.info(
                    "当前 strategy.py 没有返回诊断数据。"
                    "请确认线上代码已更新为带 diagnostics 的版本。"
                )

            # -------------------------
            # 权益曲线
            # -------------------------
            st.subheader("权益曲线")

            equity = result["equity"]

            if not equity.empty:
                st.line_chart(
                    equity.set_index("time")["equity"]
                )
            else:
                st.info("当前没有可显示的权益曲线数据。")

            # -------------------------
            # 交易记录
            # -------------------------
            st.subheader("交易记录")

            if trades:
                trade_df = pd.DataFrame(trades)

                st.dataframe(
                    trade_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "下载交易记录 CSV",
                    trade_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="kraken_btc_futures_trades.csv",
                    mime="text/csv"
                )

                if "net_pnl" in trade_df.columns:
                    st.subheader("交易盈亏分析")

                    winning_trades = int(
                        (trade_df["net_pnl"] > 0).sum()
                    )
                    losing_trades = int(
                        (trade_df["net_pnl"] < 0).sum()
                    )
                    breakeven_trades = int(
                        (trade_df["net_pnl"] == 0).sum()
                    )

                    a1, a2, a3 = st.columns(3)

                    a1.metric("盈利交易", winning_trades)
                    a2.metric("亏损交易", losing_trades)
                    a3.metric("盈亏持平", breakeven_trades)

                    with st.expander("查看交易盈亏统计"):
                        st.write(
                            "以下统计仅基于本次已结束的交易记录。"
                        )

                        st.write(
                            f"累计盈利交易数：{winning_trades}"
                        )
                        st.write(
                            f"累计亏损交易数：{losing_trades}"
                        )
                        st.write(
                            f"盈亏持平交易数：{breakeven_trades}"
                        )

            else:
                st.info(
                    "当前没有已结束的交易记录。"
                    "请先查看策略信号诊断，"
                    "判断是没有信号，还是信号被过滤。"
                )

            st.caption(
                "本原型仅用于历史研究，不连接真实下单接口。"
                "手续费、滑点和成交模型均为假设，"
                "资金费率暂未纳入，回测结果不代表未来表现。"
            )

        except Exception as e:
            st.error(
                f"运行失败：{type(e).__name__}: {e}"
            )

            st.info(
                "请复制完整错误信息发给助手，"
                "以便检查行情接口、数据格式或回测逻辑。"
            )

else:
    st.markdown("""
### 当前版本包含

- 获取 Kraken BTC 期货公开 K 线
- 1 小时环境过滤、15 分钟候选区域、5 分钟反转确认
- 固定风险预算、止损止盈、手续费和滑点假设
- 回测收益指标、权益曲线和交易记录导出
- 多头、空头候选信号及趋势、风险过滤诊断

### 尚未包含

- 历史订单簿深度回放与主动成交分析
- 历史资金费率精确结算
- 实时模拟成交与告警
- 真实下单（明确禁用）

### 数据说明

当前使用 Kraken 的 PF_XBTUSD 期货行情，
不是 Binance 的 BTCUSDT 行情。
两者的价格、合约规则和交易成本可能不同，
因此本次结果只用于验证原型和研究策略。
""")
