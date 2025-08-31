import typing
import numpy as np
import pandas as pd

from FinMind.schema.data import Dataset
from FinMind.strategies.base_sql import Strategy


class SmartDCAForBigCap(Strategy):
    """
    智慧定期定額（Smart DCA）策略，適合權值股（如 2330, 2454, 2317 等）。

    邏輯：
      - 每月固定買進
      - 價格需在 MA240 (年線) 之上才執行
      - RSI < 30 或 KD < 20 → 加碼 2 倍
      - RSI > 70 → 暫停買進
      - 若價格跌破年線 → 停止定期定額，直到重新站上
    """

    rsi_n: int = 14
    kdj_n: int = 9
    base_amount: int = 1
    use_next_day: bool = True

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.sort_values("date").copy()
        sp.index = range(len(sp))

        required_cols = ["date", "stock_id", "open", "close", "max", "min"]
        for col in required_cols:
            if col not in sp.columns:
                raise KeyError(f"Missing column '{col}' required by SmartDCAForBigCap.")

        # 計算年線 (MA240)
        sp["ma240"] = sp["close"].rolling(240, min_periods=1).mean()

        # 計算 RSI
        delta = sp["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_n, min_periods=1).mean()
        avg_loss = loss.rolling(self.rsi_n, min_periods=1).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        sp["rsi"] = 100 - (100 / (1 + rs))

        # 計算 KD (隨機指標) → 改用 min/max
        low_min = sp["min"].rolling(self.kdj_n, min_periods=1).min()
        high_max = sp["max"].rolling(self.kdj_n, min_periods=1).max()
        sp["rsv"] = (sp["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        sp["kdj_k"] = sp["rsv"].ewm(com=2).mean()
        sp["kdj_d"] = sp["kdj_k"].ewm(com=2).mean()

        # 每月只買一次（用月份變化）
        sp["month"] = pd.to_datetime(sp["date"]).dt.to_period("M")
        sp["month_changed"] = sp["month"] != sp["month"].shift(1)

        signal = np.zeros(len(sp), dtype=int)

        for i in range(len(sp)):
            if not sp.at[i, "month_changed"]:  # 只在月初操作
                continue

            price = float(sp.at[i, "close"])
            rsi = float(sp.at[i, "rsi"])
            k = float(sp.at[i, "kdj_k"])
            ma240 = float(sp.at[i, "ma240"])

            if price > ma240:  # 站上年線才操作
                if rsi > 70:  
                    continue  # 過熱 → 暫停買進
                elif (rsi < 30) or (k < 20):
                    signal[i] = self.base_amount * 2  # 超跌 → 加碼 2 倍
                else:
                    signal[i] = self.base_amount  # 正常定期定額
            else:
                continue  # 跌破年線 → 暫停

        sp["signal"] = signal

        if self.use_next_day:
            sp["signal"] = sp["signal"].shift(1).fillna(0).astype(int)

        return sp
