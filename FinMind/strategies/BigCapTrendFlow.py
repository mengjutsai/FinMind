import typing
import numpy as np
import pandas as pd

from FinMind.schema.data import Dataset
from FinMind.strategies.base_sql import Strategy


class BigCapTrendFlow(Strategy):
    """
    適合權值股的長線策略：
    入場：
      - 大盤 (TAIEX) > 季線 (MA120)
      - 個股 close > MA120 且 MA60 > MA120
      - 外資近20日累積買超 > 0
    出場：
      - 跌破 MA120
      - 或外資20日買超 < 0
      - 或停損 (預設 -10%)

    參數：
      stop_loss_pct: 停損比率 (預設 0.10 = -10%)
      use_next_day: 是否隔日才能執行 (預設 True)
    """

    stop_loss_pct: float = 0.10
    use_next_day: bool = True

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.sort_values("date").copy()
        sp.index = range(len(sp))

        required_cols = ["date", "stock_id", "close"]
        for col in required_cols:
            if col not in sp.columns:
                raise KeyError(f"Missing column '{col}' required by BigCapTrendFlow.")

        # ========== 成交量處理 ==========
        vol_col = "Trading_Volume" if "Trading_Volume" in sp.columns else (
            "TradingVolume" if "TradingVolume" in sp.columns else None
        )
        if vol_col is None:
            raise KeyError("Missing volume column ('Trading_Volume' or 'TradingVolume').")
        sp["vol"] = sp[vol_col].astype(float)

        # ========== 大盤資料 ==========
        taiex = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockPrice,
            data_id="TAIEX",
            start_date=self.start_date,
            end_date=self.end_date,
        )
        taiex = taiex[["date", "close"]].rename(columns={"close": "taiex_close"})
        taiex["taiex_ma120"] = taiex["taiex_close"].rolling(120, min_periods=1).mean()
        sp = pd.merge(sp, taiex, on="date", how="left")

        # ========== 外資資料 ==========
        inst = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockInstitutionalInvestorsBuySell,
            data_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        inst = inst[inst["name"] == "Foreign_Investor"].copy()
        inst["net"] = inst["buy"] - inst["sell"]
        inst = inst.groupby(["date", "stock_id"], as_index=False)["net"].sum()
        sp = pd.merge(sp, inst, on=["date", "stock_id"], how="left").fillna({"net": 0.0})
        sp["net20"] = sp["net"].rolling(20, min_periods=1).sum()

        # ========== 均線 ==========
        sp["ma60"] = sp["close"].rolling(60, min_periods=1).mean()
        sp["ma120"] = sp["close"].rolling(120, min_periods=1).mean()

        # ========== 入場條件 ==========
        sp["long_setup"] = (
            (sp["taiex_close"] > sp["taiex_ma120"]) &
            (sp["close"] > sp["ma120"]) &
            (sp["ma60"] > sp["ma120"]) &
            (sp["net20"] > 0)
        )

        # ========== 出場條件 ==========
        sp["exit_setup"] = (
            (sp["close"] < sp["ma120"]) |
            (sp["net20"] < 0)
        )

        # ========== 持倉邏輯 ==========
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

        # ========== 避免前視 ==========
        if self.use_next_day:
            sp["signal"] = sp["signal"].shift(1).fillna(0).astype(int)

        return sp
