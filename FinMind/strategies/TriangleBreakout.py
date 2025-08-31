import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from FinMind.strategies.base_sql import Strategy

class TriangleBreakout(Strategy):
    """
    三角收斂突破策略
    - 用最近 N 日高低點做線性回歸近似趨勢線
    - 收盤價突破上軌(高點線) 且放量 → 向上突破
    - 收盤價跌破下軌(低點線) 且放量 → 向下突破
    """

    n_days: int = 50
    vol_ratio_th: float = 1.5

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.copy()
        sp["vol_ma20"] = sp["Trading_Volume"].rolling(20, min_periods=1).mean()

        upper_line = []
        lower_line = []
        for i in range(len(sp)):
            if i < self.n_days:
                upper_line.append(np.nan)
                lower_line.append(np.nan)
                continue

            # 最近 n_days 的高點 / 低點
            highs = sp.iloc[i-self.n_days+1:i+1]["max"].values.reshape(-1,1)
            lows = sp.iloc[i-self.n_days+1:i+1]["min"].values.reshape(-1,1)
            t = np.arange(self.n_days).reshape(-1,1)

            # 高點線性回歸 (下降趨勢線)
            reg_high = LinearRegression().fit(t, highs)
            pred_high = reg_high.predict([[self.n_days-1]])[0][0]

            # 低點線性回歸 (上升趨勢線)
            reg_low = LinearRegression().fit(t, lows)
            pred_low = reg_low.predict([[self.n_days-1]])[0][0]

            upper_line.append(pred_high)
            lower_line.append(pred_low)

        sp["upper"] = upper_line
        sp["lower"] = lower_line

        # 突破條件
        sp["up_break"] = (sp["close"] > sp["upper"]) & (sp["Trading_Volume"] > sp["vol_ma20"] * self.vol_ratio_th)
        sp["down_break"] = (sp["close"] < sp["lower"]) & (sp["Trading_Volume"] > sp["vol_ma20"] * self.vol_ratio_th)

        # 產生訊號
        signal = np.zeros(len(sp), dtype=int)
        for i in range(len(sp)):
            if bool(sp.at[sp.index[i], "up_break"]):
                signal[i] = 1   # 多單
            elif bool(sp.at[sp.index[i], "down_break"]):
                signal[i] = -1  # 空單

        sp["signal"] = signal
        return sp
