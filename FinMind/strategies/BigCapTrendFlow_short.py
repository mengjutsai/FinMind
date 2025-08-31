import typing
import numpy as np
import pandas as pd

from FinMind.schema.data import Dataset
from FinMind.strategies.base_sql import Strategy


class BigCapTrendFlow(Strategy):
    """
    權值股策略：法人動能 + 趨勢結構 + 相對大盤強度

    入場（僅多單）：
      - 外資 n 日累積淨買超的 z-score > z_th
      - 股價結構多頭：close > MA60 且 MA20 > MA60
      - 個股相對大盤強勢： (stock_close / TAIEX_close) > SMA20
    出場：
      - 外資轉為顯著淨賣超（z-score < -z_exit）
      - 或 跌破 MA60
      - 或 停損（相對進場價）

    參數：
      n_days:        外資累積視窗（預設 5）
      z_th:          入場 z-score 門檻（預設 1.0）
      z_exit:        出場 z-score 門檻（預設 0.5）
      stop_loss_pct: 停損（預設 0.08 = 8%）
      use_next_day:  是否隔日執行（預設 True）
    """

    n_days: int = 5
    z_th: float = 1.0
    z_exit: float = 0.5
    stop_loss_pct: float = 0.08
    use_next_day: bool = True

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.sort_values("date").copy()
        sp.index = range(len(sp))

        # 確認必備欄位
        required_cols = ["date", "stock_id", "close"]
        for col in required_cols:
            if col not in sp.columns:
                raise KeyError(f"Missing column '{col}' required by BigCapTrendFlow.")

        # 取得外資買賣超
        inst = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockInstitutionalInvestorsBuySell,
            data_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        buy_col = "buy" if "buy" in inst.columns else "Foreign_Investors_Buy"
        sell_col = "sell" if "sell" in inst.columns else "Foreign_Investors_Sell"
        inst = inst.groupby(["date", "stock_id"], as_index=False).agg({
            buy_col: np.sum,
            sell_col: np.sum
        })
        inst["net"] = inst[buy_col] - inst[sell_col]

        sp = pd.merge(sp, inst[["date", "stock_id", "net"]],
                      on=["date", "stock_id"], how="left").fillna({"net": 0.0})

        # 均線
        sp["ma20"] = sp["close"].rolling(20, min_periods=1).mean()
        sp["ma60"] = sp["close"].rolling(60, min_periods=1).mean()

        # 外資 n 日累積 + z-score
        sp["net_n"] = sp["net"].rolling(self.n_days, min_periods=1).sum()
        mu = sp["net_n"].rolling(20, min_periods=5).mean()
        sd = sp["net_n"].rolling(20, min_periods=5).std(ddof=0)
        sp["z"] = (sp["net_n"] - mu) / (sd.replace(0, np.nan))

        # 取得大盤（TAIEX）資料
        twii = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockPrice,
            data_id="TAIEX",
            start_date=self.start_date,
            end_date=self.end_date,
        )
        twii = twii[["date", "close"]].rename(columns={"close": "twii_close"})
        sp = pd.merge(sp, twii, on="date", how="left").fillna(method="ffill")

        # 相對強弱
        sp["rel_strength"] = sp["close"] / sp["twii_close"]
        sp["rel_ma20"] = sp["rel_strength"].rolling(20, min_periods=1).mean()

        # 入場條件
        sp["long_setup"] = (
            (sp["z"] > self.z_th)
            & (sp["close"] > sp["ma60"])
            & (sp["ma20"] > sp["ma60"])
            & (sp["rel_strength"] > sp["rel_ma20"])
        )

        # 出場條件
        sp["exit_setup"] = (
            (sp["z"] < -self.z_exit)
            | (sp["close"] < sp["ma60"])
        )

        # 產生訊號
        signal = np.zeros(len(sp), dtype=int)
        holding = False
        entry_price = np.nan

        for i in range(len(sp)):
            c = float(sp.at[i, "close"])
            if not holding:
                if bool(sp.at[i, "long_setup"]):
                    signal[i] = 1
                    holding = True
                    entry_price = c
            else:
                stop_loss_hit = c <= entry_price * (1 - self.stop_loss_pct)
                if bool(sp.at[i, "exit_setup"]) or stop_loss_hit:
                    signal[i] = -1
                    holding = False
                    entry_price = np.nan

        sp["signal"] = signal

        if self.use_next_day:
            sp["signal"] = sp["signal"].shift(1).fillna(0).astype(int)

        return sp
