import streamlit as st
import pandas as pd
from strategy import fetch_klines, run_backtest

st.set_page_config(
    page_title="BTC Reversal Intelligence v1",
    layout="wide"
)

st.title("BTC Reversal Intelligence v1")
st.caption(
    "Kraken BTC期货反转策略 · 历史回测原型 · 不会发送真实订单"
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

    with st.spinner("正在获取 Kraken BTC 期货 K 线并运行回测……"):

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
                f"最新收盘价：{float(df['close'].iloc[-1]):.2f}"
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

            c4.metric(
                "盈利因子",
                "N/A" if pf is None else f"{pf:.2f}"
            )

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

            st.subheader("权益曲线")

            equity = result["equity"]

            if not equity.empty:
                st.line_chart(
                    equity.set_index("time")["equity"]
                )
            else:
                st.info("当前没有可显示的权益曲线数据。")

            st.subheader("交易记录")

            if trades:
                trade_df = pd.DataFrame(trades)

                st.dataframe(
                    trade_df,
                    use_container_width=True
                )

                st.download_button(
                    "下载交易记录 CSV",
                    trade_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="kraken_btc_futures_trades.csv",
                    mime="text/csv"
                )

            else:
                st.info(
                    "该参数组合没有产生交易。"
                    "请先检查行情数据和策略条件，"
                    "不要为了增加交易数量而随意放宽规则。"
                )

            st.caption(
                "本原型仅用于历史研究，不连接真实下单接口。"
                "手续费、滑点和成交模型均为假设，"
                "回测结果尚未验证，不代表未来表现。"
            )

        except Exception as e:
            st.error(f"运行失败：{type(e).__name__}: {e}")

            st.info(
                "请把完整错误信息发给助手，"
                "以便检查行情接口、数据格式或回测逻辑。"
            )

else:
    st.markdown("""
### 当前版本包含

- 获取 Kraken BTC 期货公开 K 线
- 1 小时环境过滤、15 分钟候选区域、5 分钟反转确认
- 固定风险预算、止损止盈、手续费和滑点假设
- 回测收益指标、权益曲线和交易记录导出
- 显示实际获取的 K 线数量、数据时间范围和最新收盘价

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
