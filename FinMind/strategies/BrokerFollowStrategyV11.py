import sqlite3
import pandas as pd
import numpy as np
from FinMind.strategies.base_sql import Strategy


class BrokerFollowStrategyV11(Strategy):
    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]
    ratio_th: float = 0.05
    zscore_th: float = 2.0
    lookback: int = 60
    stop_loss: float = 0.10          # 強制停損 10%
    trailing_stop: float = 0.05      # 移動停利 5%
    take_profit: float = 0.08        # 停利 8%
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = stock_price.drop(columns=["fee","tax"], errors="ignore")
        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"], errors="coerce")

        # === 抓籌碼 ===
        conn = sqlite3.connect(self.db_file)
        q = f"""
            SELECT date, stock_id, SUM(net) AS net
            FROM tw_trading_daily_report
            WHERE stock_id = '{self.stock_id}'
            AND securities_trader_id IN ({",".join(map(str, self.SECURITIES_TRADER_IDS))})
            AND date BETWEEN '{self.start_date}' AND '{self.end_date}'
            GROUP BY date, stock_id
            ORDER BY date
        """
        broker_df = pd.read_sql_query(q, conn, parse_dates=["date"])
        conn.close()

        if broker_df.empty:
            stock_price["signal"] = 0.0
            return stock_price

        broker_df["net_lots"] = broker_df["net"] / 1000.0
        merged = stock_price.merge(
            broker_df[["date", "net_lots"]], on="date", how="left"
        ).fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # === Z-score ===
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 狀態變數 ===
        merged["signal"] = 0.0
        position_size = 0.0
        avg_entry_price = None
        max_entry_price = None
        peak_price = None

        for i in range(len(merged)):
            row = merged.iloc[i]
            price_now = row["close"]

            # === 出場條件 ===
            if position_size > 0:
                # 強制停損 (跌 10%)
                if avg_entry_price and price_now <= avg_entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -position_size
                    position_size = 0.0
                    avg_entry_price = max_entry_price = peak_price = None
                    continue

                # 部分停利 (每次達到都賣一半)
                if avg_entry_price and price_now >= avg_entry_price * (1 + self.take_profit):
                    sell_lots = max(1.0, position_size / 2)
                    position_size -= sell_lots
                    merged.loc[i, "signal"] = -sell_lots
                    if position_size == 0:
                        avg_entry_price = max_entry_price = peak_price = None
                    continue

                # 移動停利 (回落 5%)
                peak_price = max(peak_price, price_now) if peak_price else price_now
                if peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -position_size
                    position_size = 0.0
                    avg_entry_price = max_entry_price = peak_price = None
                    continue

                # 籌碼反轉 (連三日賣超)
                if i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -position_size
                    position_size = 0.0
                    avg_entry_price = max_entry_price = peak_price = None
                    continue

            # === 進場 / 加倉判斷 ===
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)
                zscore_score = min(1.0, row["zscore"] / 5.0)
                strength = 0.6 * ratio_score + 0.4 * zscore_score
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉，不減倉
                    if avg_entry_price is None:
                        avg_entry_price = row["close"]
                        max_entry_price = row["close"]
                    else:
                        total_value = avg_entry_price * position_size + row["close"] * (new_position - position_size)
                        avg_entry_price = total_value / new_position
                        max_entry_price = max(max_entry_price, row["close"])

                    position_size = new_position
                    peak_price = row["close"] if peak_price is None else max(peak_price, row["close"])

                merged.loc[i, "signal"] = position_size
            else:
                merged.loc[i, "signal"] = position_size

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged

