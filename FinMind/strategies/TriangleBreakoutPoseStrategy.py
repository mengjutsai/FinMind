import pandas as pd
import numpy as np
from FinMind.strategies.base_sql import Strategy

class TriangleBreakoutPoseStrategy(Strategy):
    """
    Pose-based Triangle Breakout Strategy (p1 左上, p2 右上, p3 右下, p4 左下)
    - p2 = 突破點 (右上)
    - 基準高度 = (p3.y + p4.y) / 2
    - 目標價 = 2*p2.y - 基準高度
    - 突破進場，達標停利，跌破支撐停損
    """

    max_hold_days: int = 20  # 最多持有天數

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.copy().reset_index(drop=True)
        sp["date"] = pd.to_datetime(sp["date"])
        sp["signal"] = 0

        # pose_events: list of dicts [{"date":..., "p1":(x,y), "p2":(x,y), "p3":(x,y), "p4":(x,y)}]
        pose_events = kwargs.get("pose_events", [])

        for evt in pose_events:
            d = pd.to_datetime(evt["date"])
            if d not in sp["date"].values:
                continue

            p1, p2, p3, p4 = evt["p1"], evt["p2"], evt["p3"], evt["p4"]

            # 基準高度 = 下邊界平均
            base_height = (p3[1] + p4[1]) / 2
            # 目標價
            target_price = 2 * p2[1] - base_height

            # 找當天收盤價
            idx = sp.index[sp["date"] == d][0]
            close_price = sp.loc[idx, "close"]

            # === 進場 ===
            if close_price > p2[1]:  # 突破右上邊界
                sp.loc[idx, "signal"] = 1  # buy signal

                # === 出場檢查 ===
                for j in range(idx+1, min(idx+1+self.max_hold_days, len(sp))):
                    future_close = sp.loc[j, "close"]

                    # 1. 達到目標 → 停利
                    if future_close >= target_price:
                        sp.loc[j, "signal"] = -1
                        break

                    # 2. 跌破右下支撐 → 停損
                    if future_close < p3[1]:
                        sp.loc[j, "signal"] = -1
                        break

        return sp
