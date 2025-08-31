import numpy as np
import pandas as pd
from FinMind.strategies.base_sql import Strategy

class TriangleBreakout_V2(Strategy):
    """
    三角收斂突破策略（改良版）
    - 用布林帶收斂判斷
    - 收盤價突破上/下軌時進場
    - 停損：回到整理區間
    - 停利：區間高度
    """

    n_days: int = 20
    vol_ratio_th: float = 1.3
    bb_width_th: float = 0.1   # 布林帶寬度閾值 (10%)
    stop_loss_pct: float = 0.05

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.copy()

        # 20 日均線與標準差
        sp["ma20"] = sp["close"].rolling(self.n_days).mean()
        sp["std20"] = sp["close"].rolling(self.n_days).std()
        sp["upper"] = sp["ma20"] + 2 * sp["std20"]
        sp["lower"] = sp["ma20"] - 2 * sp["std20"]

        # 布林帶寬度（相對於均價）
        sp["bb_width"] = (sp["upper"] - sp["lower"]) / sp["ma20"]

        # 最近 N 日高低點（決定停利目標）
        sp["hhv"] = sp["close"].rolling(self.n_days).max()
        sp["llv"] = sp["close"].rolling(self.n_days).min()
        sp["triangle_height"] = sp["hhv"] - sp["llv"]

        # 成交量均值（避免重複算）
        sp["vol_ma20"] = sp["Trading_Volume"].rolling(20, min_periods=1).mean()

        # 產生訊號
        signal = np.zeros(len(sp), dtype=int)
        entry_price = np.nan
        holding = False
        target_price, stop_loss = np.nan, np.nan

        for i in range(len(sp)):
            c = sp.at[i, "close"]
            vol = sp.at[i, "Trading_Volume"]

            if not holding:
                # 收斂區間內才考慮進場
                if sp.at[i, "bb_width"] < self.bb_width_th:
                    # 上突破 → 做多
                    if c > sp.at[i, "upper"] and vol > sp.at[i, "vol_ma20"] * self.vol_ratio_th:
                        signal[i] = 1
                        holding = True
                        entry_price = c
                        target_price = c + sp.at[i, "triangle_height"]
                        stop_loss = c * (1 - self.stop_loss_pct)
                    # 下突破 → 做空
                    elif c < sp.at[i, "lower"] and vol > sp.at[i, "vol_ma20"] * self.vol_ratio_th:
                        signal[i] = -1
                        holding = True
                        entry_price = c
                        target_price = c - sp.at[i, "triangle_height"]
                        stop_loss = c * (1 + self.stop_loss_pct)

            else:
                # 出場條件：達停利 or 停損
                if signal[i-1] == 1:  # 多單
                    if c >= target_price or c <= stop_loss:
                        signal[i] = -1
                        holding = False
                        entry_price = np.nan
                elif signal[i-1] == -1:  # 空單
                    if c <= target_price or c >= stop_loss:
                        signal[i] = 1
                        holding = False
                        entry_price = np.nan

        sp["signal"] = signal
        return sp
