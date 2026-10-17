import streamlit as st
import pandas as pd
from strategy import fetch_klines, run_backtest

st.set_page_config(page_title="BTC Reversal Intelligence v1", layout="wide")
st.title("BTC Reversal Intelligence v1")
st.caption("BTCUSDT 永续反转策略 · 历史回测原型 · 不会发送真实订单")

with st.sidebar:
    st.header("回测设置")
    days = st.selectbox("历史数据范围", [7, 14, 30, 60, 90], index=2)
    capital = st.number_input("模拟初始资金 (USDT)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("单笔风险预算 (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    fee_bps = st.number_input("单边手续费 (bps)", min_value=0.0, value=5.0, step=0.5)
    slippage_bps = st.number_input("单边滑点假设 (bps)", min_value=0.0, value=2.0, step=0.5)
    min_rr = st.number_input("最低目标盈亏比", min_value=1.0, max_value=5.0, value=2.0, step=0.25)

st.warning("原型限制：回测使用公开K线，不包含历史订单簿重放；资金费率暂未纳入；回测结果不代表未来表现。")
if st.button("获取行情并运行回测", type="primary"):
    with st.spinner("正在获取 Binance BTCUSDT 5分钟K线并回测……"):
        try:
            df = fetch_klines(days=days)
            result = run_backtest(
                df, initial_capital=capital, risk_pct=risk_pct / 100,
                fee_bps=fee_bps, slippage_bps=slippage_bps, min_rr=min_rr
            )
            trades = result["trades"]
            m = result["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("净收益", f'{m["net_pnl"]:.2f} USDT', f'{m["return_pct"]:.2f}%')
            c2.metric("交易笔数", str(m["trades"]))
            c3.metric("胜率", f'{m["win_rate"]:.1f}%')
            c4.metric("盈利因子", "N/A" if m["profit_factor"] is None else f'{m["profit_factor"]:.2f}')
            c5, c6, c7 = st.columns(3)
            c5.metric("最大回撤", f'{m["max_drawdown_pct"]:.2f}%')
            c6.metric("平均每笔净收益", f'{m["avg_trade"]:.2f} USDT')
            c7.metric("期末权益", f'{m["final_equity"]:.2f} USDT')

            st.subheader("权益曲线")
            if not result["equity"].empty:
                st.line_chart(result["equity"].set_index("time")["equity"])
            st.subheader("交易记录")
            if trades:
                tdf = pd.DataFrame(trades)
                st.dataframe(tdf, use_container_width=True)
                st.download_button("下载交易记录 CSV", tdf.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="btc_reversal_trades.csv", mime="text/csv")
            else:
                st.info("该参数组合没有产生交易。不要为了增加交易数量而随意放宽规则；先检查数据和策略条件。")
            st.caption("注意：该原型的信号和成交模型仅供研究，尚未经过验证，不连接任何真实下单接口。")
        except Exception as e:
            st.error(f"运行失败：{e}")
            st.info("请检查网络连接、Binance API可用性及依赖安装。")
else:
    st.markdown("""
### 当前版本包含
- 拉取公开 BTCUSDT 永续 5 分钟 K 线
- 1小时环境过滤、15分钟候选区域、5分钟反转确认
- 固定风险预算、止损止盈、费用和滑点假设
- 回测指标、权益曲线和交易记录导出

### 尚未包含
- 历史订单簿深度回放与主动成交分析
- 历史资金费率精确结算
- 实时模拟成交与告警
- 真实下单（明确禁用）
""")
