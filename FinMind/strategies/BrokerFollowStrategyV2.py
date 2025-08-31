import sqlite3
import pandas as pd
import numpy as np
from FinMind.strategies.base_sql import Strategy

class BrokerFollowStrategyV2(Strategy):
    """
    策略概念: 跟單主力券商分點
    規則:
        - 單日或連續買超達門檻 → 買進
        - 出場條件: 賣超/停利/停損/籌碼反轉
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    buy_th: int = 1000          # 進場門檻 (張)
    sell_th: int = 5000         # 單日賣超門檻 (張)
    consecutive_days: int = 2   # 連續買超天數
    take_profit: float = 0.15   # 停利 15%
    stop_loss: float = 0.08     # 停損 8%
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"])

        # 從 SQLite 抓籌碼
        conn = sqlite3.connect(self.db_file)
        q = f"""
            SELECT date, stock_id,
                SUM(total_buy) AS buy,
                SUM(total_sell) AS sell,
                SUM(net) AS net
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
            stock_price["signal"] = 0
            return stock_price

        broker_df["net_lots"] = broker_df["net"] / 1000.0

        # 合併到股價
        stock_price = stock_price.merge(
            broker_df[["date", "net_lots"]],
            on="date", how="left"
        ).fillna(0)

        # === 初始化 ===
        stock_price["signal"] = 0
        entry_price = None

        for i in range(len(stock_price)):
            row = stock_price.iloc[i]

            # 進場條件：連續買超 or 單日大買
            if (
                row["net_lots"] > self.buy_th or
                (i >= self.consecutive_days and
                 (stock_price["net_lots"].iloc[i-self.consecutive_days+1:i+1] > self.buy_th).all())
            ):
                stock_price.loc[i, "signal"] = 1
                entry_price = row["close"]

            # 出場條件
            elif entry_price:
                price_now = row["close"]

                # 停利
                if price_now >= entry_price * (1 + self.take_profit):
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None

                # 停損
                elif price_now <= entry_price * (1 - self.stop_loss):
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None

                # 單日大賣
                elif row["net_lots"] < -self.sell_th:
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None

                # 籌碼反轉：連續三日賣超
                elif i >= 3 and (stock_price["net_lots"].iloc[i-2:i+1] < 0).all():
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None

        stock_price["date"] = stock_price["date"].dt.strftime("%Y-%m-%d")
        return stock_price
