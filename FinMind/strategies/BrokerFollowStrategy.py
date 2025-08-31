import sqlite3
import pandas as pd
import numpy as np
from FinMind.strategies.base_sql import Strategy

class BrokerFollowStrategy(Strategy):
    """
    策略概念: 特定券商群組若持續買超，可能推升股價；若大幅賣超，可能壓低股價
    策略規則:
        - SECURITIES_TRADER_IDS 券商群組
        - 合併群組的淨買超 (張)
        - 若連續 n 天淨買超 > buy_th → 發出買進訊號
        - 若單日淨賣超 < -sell_th → 發出賣出訊號
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    buy_th: int = 1000          # 進場門檻 (張)
    sell_th: int = 10000         # 出場門檻 (張)
    consecutive_days: int = 3  # 連續買超天數
    db_file: str = "stock.db"  # SQLite DB

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = stock_price.sort_values("date").reset_index(drop=True)

        # 轉成 datetime
        stock_price["date"] = pd.to_datetime(stock_price["date"])

        # 從 SQLite 抓取多券商資料
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

        # 換算成張
        broker_df["net_lots"] = broker_df["net"] / 1000.0

        # 合併
        stock_price = stock_price.merge(
            broker_df[["date", "net_lots"]],
            on="date", how="left"
        ).fillna(0)

        # 產生訊號
        stock_price["signal"] = 0
        stock_price["consecutive_buy"] = (
            stock_price["net_lots"]
            .rolling(self.consecutive_days)
            .apply(lambda x: (x > self.buy_th).all(), raw=True)
        )
        stock_price.loc[stock_price["consecutive_buy"] == 1, "signal"] = 1
        stock_price.loc[stock_price["net_lots"] < -self.sell_th, "signal"] = -1

        stock_price["date"] = stock_price["date"].dt.strftime("%Y-%m-%d")

        return stock_price
