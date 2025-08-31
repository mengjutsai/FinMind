import typing
import numpy as np
import pandas as pd
from FinMind.schema.data import Dataset
from FinMind.strategies.base_sql import Strategy

class InstitutionalInvestorsFollower_V2(Strategy):
    """
    改良版：法人策略 + 停損/停利 + 外資賣超出場
    """

    n_days = 10
    stop_loss = 0.05      # 5% 停損
    take_profit = 0.1     # 10% 停利
    max_hold_days = 20    # 最長持有天數

    def create_trade_sign(
        self, stock_price: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        stock_price = stock_price.sort_values("date").reset_index(drop=True)

        # === 取得外資買賣超 ===
        institutional_investors_buy_sell = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockInstitutionalInvestorsBuySell,
            data_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        institutional_investors_buy_sell = (
            institutional_investors_buy_sell.groupby(
                ["date", "stock_id"], as_index=False
            ).agg({"buy": np.sum, "sell": np.sum})
        )
        institutional_investors_buy_sell["diff"] = (
            institutional_investors_buy_sell["buy"]
            - institutional_investors_buy_sell["sell"]
        )
        stock_price = pd.merge(
            stock_price,
            institutional_investors_buy_sell[["stock_id", "date", "diff"]],
            on=["stock_id", "date"],
            how="left",
        ).fillna(0)

        # === 外資異常訊號 ===
        stock_price["signal_info"] = self.detect_Abnormal_Peak(
            y=stock_price["diff"].values,
            lag=self.n_days,
            threshold=3,
            influence=0.35,
        )
        stock_price["signal"] = 0

        # === 停利/停損/外資大賣機制 ===
        holding = False
        entry_price = 0
        entry_day = 0

        for i in range(len(stock_price)):
            if not holding:
                # 進場判斷
                if stock_price.loc[i, "signal_info"] == -1:  # 外資買超 → 進場
                    stock_price.loc[i, "signal"] = 1
                    holding = True
                    entry_price = stock_price.loc[i, "close"]
                    entry_day = i
                elif stock_price.loc[i, "signal_info"] == 1:  # 外資賣超 → 做空
                    stock_price.loc[i, "signal"] = -1
            else:
                # 已進場 → 檢查出場條件
                pct_change = (stock_price.loc[i, "close"] - entry_price) / entry_price
                hold_days = i - entry_day

                # 停損
                if pct_change <= -self.stop_loss:
                    stock_price.loc[i, "signal"] = -1
                    holding = False

                # 停利
                elif pct_change >= self.take_profit:
                    stock_price.loc[i, "signal"] = -1
                    holding = False

                # 持倉時間過久
                elif hold_days >= self.max_hold_days:
                    stock_price.loc[i, "signal"] = -1
                    holding = False

                # 新增：外資大賣超，強制出場
                elif stock_price.loc[i, "signal_info"] == 1:
                    stock_price.loc[i, "signal"] = -1
                    holding = False

        return stock_price

    def detect_Abnormal_Peak(
        self, y: np.array, lag: int, threshold: float, influence: float
    ) -> typing.List[float]:
        signals = np.zeros(len(y))
        filtered_y = np.array(y)
        avg_filter = [0] * len(y)
        std_filter = [0] * len(y)
        avg_filter[lag - 1] = np.mean(y[0:lag])
        std_filter[lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(y)):
            if abs(y[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
                signals[i] = 1 if y[i] > avg_filter[i - 1] else -1
                filtered_y[i] = (
                    influence * y[i] + (1 - influence) * filtered_y[i - 1]
                )
            else:
                signals[i] = 0
                filtered_y[i] = y[i]
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1): i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1): i + 1])
        return list(signals)
