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



class BrokerFollowStrategyV3(Strategy):
    """
    策略概念: 跟單主力券商分點 (進階版)
    - 進場條件: 主力買超 + 技術過濾 (均線/量能/多券商共振)
    - 出場條件: 停利/停損/籌碼反轉/技術反轉
    - 風控: 部位加減碼、動態停損
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    buy_th: int = 1000          # 單日進場門檻 (張)
    sell_th: int = 5000         # 單日賣超門檻 (張)
    consecutive_days: int = 2   # 連續買超天數
    take_profit: float = 0.15   # 停利 15%
    stop_loss: float = 0.08     # 停損 8%
    trailing_stop: float = 0.05 # 移動停利 (最高價回檔 5%)
    ma_window: int = 20         # 均線過濾
    vol_window: int = 20        # 量能過濾
    min_broker_count: int = 3   # 至少幾家外資同時買超
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"])

        # 從 SQLite 抓籌碼
        conn = sqlite3.connect(self.db_file)
        q = f"""
            SELECT date, stock_id, securities_trader_id,
                total_buy, total_sell, net
            FROM tw_trading_daily_report
            WHERE stock_id = '{self.stock_id}'
            AND securities_trader_id IN ({",".join(map(str, self.SECURITIES_TRADER_IDS))})
            AND date BETWEEN '{self.start_date}' AND '{self.end_date}'
            ORDER BY date
        """
        broker_df = pd.read_sql_query(q, conn, parse_dates=["date"])
        conn.close()

        if broker_df.empty:
            stock_price["signal"] = 0
            return stock_price

        # 聚合: 每日總和 + 買超券商家數
        agg_df = broker_df.groupby("date").agg(
            buy=("total_buy", "sum"),
            sell=("total_sell", "sum"),
            net=("net", "sum"),
            broker_buy_count=("net", lambda x: (x > self.buy_th).sum())
        ).reset_index()
        agg_df["net_lots"] = agg_df["net"] / 1000.0

        # 合併到股價
        stock_price = stock_price.merge(
            agg_df[["date", "net_lots", "broker_buy_count"]],
            on="date", how="left"
        ).fillna(0)

        # 技術過濾
        stock_price["ma"] = stock_price["close"].rolling(self.ma_window).mean()
        stock_price["vol_ma"] = stock_price["Trading_Volume"].rolling(self.vol_window).mean()

        stock_price["signal"] = 0
        entry_price = None
        peak_price = None

        for i in range(len(stock_price)):
            row = stock_price.iloc[i]

            # === 進場條件 ===
            if (
                row["net_lots"] > self.buy_th or
                (i >= self.consecutive_days and
                 (stock_price["net_lots"].iloc[i-self.consecutive_days+1:i+1] > self.buy_th).all())
            ):
                # 技術過濾: 股價在均線之上 + 當日量大於均量 + 至少多家外資同買
                if (
                    row["close"] > row["ma"] and
                    row["Trading_Volume"] > row["vol_ma"] and
                    row["broker_buy_count"] >= self.min_broker_count
                ):
                    stock_price.loc[i, "signal"] = 1
                    entry_price = row["close"]
                    peak_price = row["close"]

            # === 出場條件 ===
            elif entry_price:
                price_now = row["close"]

                # 更新最高價
                if peak_price is not None:
                    peak_price = max(peak_price, price_now)

                # 停利 (固定目標)
                if price_now >= entry_price * (1 + self.take_profit):
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None
                    peak_price = None

                # 移動停利
                elif peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None
                    peak_price = None

                # 停損
                elif price_now <= entry_price * (1 - self.stop_loss):
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None
                    peak_price = None

                # 單日大賣
                elif row["net_lots"] < -self.sell_th:
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None
                    peak_price = None

                # 籌碼反轉：連續三日賣超
                elif i >= 3 and (stock_price["net_lots"].iloc[i-2:i+1] < 0).all():
                    stock_price.loc[i, "signal"] = -1
                    entry_price = None
                    peak_price = None

        stock_price["date"] = stock_price["date"].dt.strftime("%Y-%m-%d")
        return stock_price


class BrokerFollowStrategyV4(Strategy):
    """
    策略概念: 外資分點跟單 (進階版 V4)
    - 進場: 主力買超佔比 + Z-score 雙過濾
    - 出場: 停利 / 停損 / 移動停利 / 籌碼反轉
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    ratio_th: float = 0.05      # 主力買超佔成交量比例門檻 (5%)
    zscore_th: float = 2.0      # 買超異常 Z-score 門檻
    lookback: int = 60          # 計算 Z-score 的回溯天數
    take_profit: float = 0.15   # 停利 15%
    stop_loss: float = 0.08     # 停損 8%
    trailing_stop: float = 0.05 # 移動停利 5%
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"])

        # 從 SQLite 抓券商籌碼
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
            stock_price["signal"] = 0
            return stock_price

        # 換算張數 & 佔比
        broker_df["net_lots"] = broker_df["net"] / 1000.0
        merged = stock_price.merge(broker_df[["date", "net_lots"]], on="date", how="left").fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # 計算 Z-score
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 交易邏輯 ===
        merged["signal"] = 0
        entry_price = None
        peak_price = None

        for i in range(len(merged)):
            row = merged.iloc[i]

            # 進場條件: 比例 + Z-score 雙過濾
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                merged.loc[i, "signal"] = 1
                entry_price = row["close"]
                peak_price = row["close"]

            # 出場條件
            elif entry_price:
                price_now = row["close"]

                if peak_price is not None:
                    peak_price = max(peak_price, price_now)

                # 停利
                if price_now >= entry_price * (1 + self.take_profit):
                    merged.loc[i, "signal"] = -1
                    entry_price, peak_price = None, None

                # 移動停利
                elif peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -1
                    entry_price, peak_price = None, None

                # 停損
                elif price_now <= entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -1
                    entry_price, peak_price = None, None

                # 籌碼反轉：連續三日淨賣超
                elif i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -1
                    entry_price, peak_price = None, None

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged



class BrokerFollowStrategyV6(Strategy):
    """
    策略概念: 外資分點跟單 (動態加倉 + 強制清倉 + 防呆版)
    - 進場: 主力買超佔比 + Z-score
    - 倉位: 隨訊號強度動態加倉 (最高 5 倉)
    - 出場: 停利 / 停損 / 移動停利 / 籌碼反轉 / 重大警訊全清
    - 防呆: 日期型別統一、忽略 fee/tax 欄位
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    ratio_th: float = 0.05      # 主力買超佔成交量比例門檻 (5%)
    zscore_th: float = 2.0      # 買超異常 Z-score 門檻
    lookback: int = 60          # Z-score 回溯天數
    take_profit: float = 0.15   # 停利 15%
    stop_loss: float = 0.08     # 停損 8%
    trailing_stop: float = 0.05 # 移動停利 5%
    major_warning_ratio: float = -0.1  # 重大警訊: 主力賣超佔成交量 < -10%
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # === 防呆：刪除 fee/tax 欄位 ===


        stock_price = stock_price.sort_values("date").reset_index(drop=True)

        # 日期型別統一
        stock_price["date"] = pd.to_datetime(stock_price["date"], errors="coerce")

        # 從 SQLite 抓券商籌碼
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

        # 防呆：刪除 fee/tax 欄位


        # 日期型別統一
        broker_df["date"] = pd.to_datetime(broker_df["date"], errors="coerce")

        # 計算籌碼指標
        broker_df["net_lots"] = broker_df["net"] / 1000.0
        merged = stock_price.merge(broker_df[["date", "net_lots"]], on="date", how="left").fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # Z-score
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 交易邏輯 ===
        merged["signal"] = 0.0   # 倉位大小 (0 ~ 5 倉)，-999 = 強制清倉
        position_size = 0.0
        entry_price = None
        peak_price = None

        for i in range(len(merged)):
            row = merged.iloc[i]

            # === 重大警訊檢查 ===
            if row["broker_ratio"] < self.major_warning_ratio:
                merged.loc[i, "signal"] = -999.0  # 全部清倉
                position_size = 0.0
                entry_price, peak_price = None, None
                continue

            # === 進場/加倉條件 ===
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                # 訊號強度 (0 ~ 1)
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)   # 20% 滿分
                zscore_score = min(1.0, row["zscore"] / 5.0)        # Z=5 滿分
                strength = 0.6 * ratio_score + 0.4 * zscore_score

                # 對應倉位: 0.5 ~ 5 倉
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉，不減倉
                    position_size = new_position
                    entry_price = row["close"] if entry_price is None else entry_price
                    peak_price = row["close"] if peak_price is None else max(peak_price, row["close"])

                merged.loc[i, "signal"] = position_size

            # === 出場條件 ===
            elif position_size > 0:
                price_now = row["close"]
                peak_price = max(peak_price, price_now) if peak_price else price_now

                # 停利
                if price_now >= entry_price * (1 + self.take_profit):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None

                # 移動停利
                elif peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None

                # 停損
                elif price_now <= entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None

                # 籌碼反轉：連續三日賣超
                elif i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None

                else:
                    merged.loc[i, "signal"] = position_size  # 持倉不變

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged




class BrokerFollowStrategyV7(Strategy):
    """
    策略概念: 外資分點跟單 (分批停利 + 移動停利)
    - 進場: 主力買超佔比 + Z-score
    - 倉位: 隨訊號強度動態加倉 (最高 5 倉)
    - 出場: 分批停利 (+8% 出一半) + 移動停利 (最高價回落 5%) + 停損
    - 防呆: 日期型別統一、忽略 fee/tax 欄位
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    ratio_th: float = 0.05      # 主力買超佔成交量比例門檻 (5%)
    zscore_th: float = 2.0      # 買超異常 Z-score 門檻
    lookback: int = 60          # Z-score 回溯天數
    stop_loss: float = 0.08     # 停損 8%
    trailing_stop: float = 0.05 # 移動停利 5%
    first_take_profit: float = 0.08  # 第一檔停利 8% 出一半
    major_warning_ratio: float = -0.1  # 重大警訊: 主力賣超佔成交量 < -10%
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # === 防呆：刪除 fee/tax 欄位 ===


        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"], errors="coerce")

        # 從 SQLite 抓券商籌碼
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


        broker_df["date"] = pd.to_datetime(broker_df["date"], errors="coerce")

        # 計算籌碼指標
        broker_df["net_lots"] = broker_df["net"] / 1000.0
        merged = stock_price.merge(broker_df[["date", "net_lots"]], on="date", how="left").fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # Z-score
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 交易邏輯 ===
        merged["signal"] = 0.0   # 倉位大小 (0 ~ 5 倉)，-999 = 強制清倉
        position_size = 0.0
        entry_price = None
        peak_price = None
        first_take_profit_done = False  # 是否已經完成第一檔停利

        for i in range(len(merged)):
            row = merged.iloc[i]

            # === 重大警訊檢查 ===
            if row["broker_ratio"] < self.major_warning_ratio:
                merged.loc[i, "signal"] = -999.0
                position_size = 0.0
                entry_price, peak_price = None, None
                first_take_profit_done = False
                continue

            # === 進場/加倉條件 ===
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)
                zscore_score = min(1.0, row["zscore"] / 5.0)
                strength = 0.6 * ratio_score + 0.4 * zscore_score
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉，不減倉
                    position_size = new_position
                    entry_price = row["close"] if entry_price is None else entry_price
                    peak_price = row["close"] if peak_price is None else max(peak_price, row["close"])

                merged.loc[i, "signal"] = position_size

            # === 出場條件 ===
            elif position_size > 0:
                price_now = row["close"]
                peak_price = max(peak_price, price_now) if peak_price else price_now

                # 第一檔停利 (出一半倉位)
                if (not first_take_profit_done) and price_now >= entry_price * (1 + self.first_take_profit):
                    position_size = position_size / 2
                    merged.loc[i, "signal"] = position_size
                    first_take_profit_done = True

                # 移動停利
                elif peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None
                    first_take_profit_done = False

                # 停損
                elif price_now <= entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None
                    first_take_profit_done = False

                # 籌碼反轉：連續三日賣超
                elif i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None
                    first_take_profit_done = False

                else:
                    merged.loc[i, "signal"] = position_size

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged



class BrokerFollowStrategyV8(Strategy):
    """
    策略概念: 外資分點跟單 (動態停損 + 分批停利 + 移動停利)
    - 進場: 主力買超佔比 + Z-score
    - 倉位: 隨訊號強度動態加倉 (最高 5 倉)
    - 出場: 停損 (以持倉成本為基準)、分批停利 (+8%)、移動停利 (回落5%)、籌碼反轉
    - 防呆: 日期型別統一、忽略 fee/tax 欄位
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]  # 美林、摩根士丹利、高盛、瑞銀、摩根大通
    ratio_th: float = 0.05      # 主力買超佔成交量比例門檻 (5%)
    zscore_th: float = 2.0      # 買超異常 Z-score 門檻
    lookback: int = 60          # Z-score 回溯天數
    stop_loss: float = 0.08     # 停損 8%
    trailing_stop: float = 0.05 # 移動停利 5%
    first_take_profit: float = 0.08  # 第一檔停利 8% 出一半
    major_warning_ratio: float = -0.1  # 重大警訊: 主力賣超佔成交量 < -10%
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # === 防呆：刪除 fee/tax 欄位 ===
        stock_price = stock_price.drop(columns=["fee","tax"], errors="ignore")

        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"], errors="coerce")

        # 從 SQLite 抓券商籌碼
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

        broker_df = broker_df.drop(columns=["fee","tax"], errors="ignore")
        broker_df["date"] = pd.to_datetime(broker_df["date"], errors="coerce")

        # 計算籌碼指標
        broker_df["net_lots"] = broker_df["net"] / 1000.0
        merged = stock_price.merge(broker_df[["date", "net_lots"]], on="date", how="left").fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # Z-score
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 交易邏輯 ===
        merged["signal"] = 0.0   # 倉位大小 (0 ~ 5 倉)，-999 = 強制清倉
        position_size = 0.0
        entry_price = None
        peak_price = None
        first_take_profit_done = False

        for i in range(len(merged)):
            row = merged.iloc[i]
            price_now = row["close"]

            # === 出場條件 (優先檢查) ===
            if position_size > 0:
                # 停損 (以持倉成本為基準)
                if price_now <= merged.loc[i, "hold_cost"] * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None
                    first_take_profit_done = False
                    continue

                # 第一檔停利
                if (not first_take_profit_done) and price_now >= entry_price * (1 + self.first_take_profit):
                    position_size = position_size / 2
                    merged.loc[i, "signal"] = position_size
                    first_take_profit_done = True
                    continue

                # 移動停利
                peak_price = max(peak_price, price_now) if peak_price else price_now
                if peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None
                    first_take_profit_done = False
                    continue

                # 籌碼反轉：連續三日賣超
                if i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    entry_price, peak_price = None, None
                    first_take_profit_done = False
                    continue

            # === 進場/加倉條件 ===
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)
                zscore_score = min(1.0, row["zscore"] / 5.0)
                strength = 0.6 * ratio_score + 0.4 * zscore_score
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉，不減倉
                    position_size = new_position
                    entry_price = row["close"] if entry_price is None else entry_price
                    peak_price = row["close"] if peak_price is None else max(peak_price, row["close"])

                merged.loc[i, "signal"] = position_size

            else:
                merged.loc[i, "signal"] = position_size

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged




class BrokerFollowStrategyV9(Strategy):
    """
    策略概念: 外資分點跟單 (內建 avg_entry_price 模擬持倉成本)
    - 進場: 主力買超佔比 + Z-score
    - 倉位: 隨訊號強度動態加倉 (最高 5 倉)
    - 出場: 停損 (以 avg_entry_price 為基準)、分批停利 (+8%)、移動停利 (回落5%)、籌碼反轉
    - avg_entry_price: 策略內部維護的加權成本 (不依賴 Trader)
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]
    ratio_th: float = 0.05
    zscore_th: float = 2.0
    lookback: int = 60
    stop_loss: float = 0.08
    trailing_stop: float = 0.05
    first_take_profit: float = 0.08
    major_warning_ratio: float = -0.1
    db_file: str = "stock.db"

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = stock_price.drop(columns=["fee","tax"], errors="ignore")
        stock_price = stock_price.sort_values("date").reset_index(drop=True)
        stock_price["date"] = pd.to_datetime(stock_price["date"], errors="coerce")

        # 抓籌碼
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

        broker_df["date"] = pd.to_datetime(broker_df["date"], errors="coerce")
        broker_df["net_lots"] = broker_df["net"] / 1000.0

        merged = stock_price.merge(broker_df[["date", "net_lots"]], on="date", how="left").fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # Z-score
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 交易邏輯 ===
        merged["signal"] = 0.0
        position_size = 0.0
        avg_entry_price = None
        peak_price = None
        first_take_profit_done = False

        for i in range(len(merged)):
            row = merged.iloc[i]
            price_now = row["close"]

            # === 出場判斷 ===
            if position_size > 0:
                # 停損 (以 avg_entry_price 為基準)
                if avg_entry_price and price_now <= avg_entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    avg_entry_price, peak_price = None, None
                    first_take_profit_done = False
                    continue

                # 第一檔停利
                if (not first_take_profit_done) and avg_entry_price and price_now >= avg_entry_price * (1 + self.first_take_profit):
                    position_size = position_size / 2
                    merged.loc[i, "signal"] = position_size
                    first_take_profit_done = True
                    continue

                # 移動停利
                peak_price = max(peak_price, price_now) if peak_price else price_now
                if peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    avg_entry_price, peak_price = None, None
                    first_take_profit_done = False
                    continue

                # 籌碼反轉
                if i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    avg_entry_price, peak_price = None, None
                    first_take_profit_done = False
                    continue

            # === 進場 / 加倉判斷 ===
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)
                zscore_score = min(1.0, row["zscore"] / 5.0)
                strength = 0.6 * ratio_score + 0.4 * zscore_score
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉，不減倉
                    # 更新加權成本
                    if avg_entry_price is None:
                        avg_entry_price = row["close"]
                    else:
                        total_value = avg_entry_price * position_size + row["close"] * (new_position - position_size)
                        avg_entry_price = total_value / new_position

                    position_size = new_position
                    peak_price = row["close"] if peak_price is None else max(peak_price, row["close"])

                merged.loc[i, "signal"] = position_size
            else:
                merged.loc[i, "signal"] = position_size

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged




class BrokerFollowStrategyV10(Strategy):
    """
    策略概念: 外資分點跟單 (avg_entry_price + hard stop)
    - 進場: 主力買超佔比 + Z-score
    - 倉位: 隨訊號強度動態加倉 (最高 5 倉)
    - 出場: 
        1) 停損 (用 max_entry_price 當基準，不被攤平)
        2) 第一檔停利 (+8% 減半)
        3) 移動停利 (回落 5%)
        4) 籌碼反轉 (連三日賣超)
    """

    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]
    ratio_th: float = 0.05
    zscore_th: float = 2.0
    lookback: int = 60
    stop_loss: float = 0.08
    trailing_stop: float = 0.05
    first_take_profit: float = 0.08
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

        broker_df["date"] = pd.to_datetime(broker_df["date"], errors="coerce")
        broker_df["net_lots"] = broker_df["net"] / 1000.0


        stock_price["date"] = pd.to_datetime(stock_price["date"], errors="coerce")
        broker_df["date"] = pd.to_datetime(broker_df["date"], errors="coerce")

        merged = stock_price.merge(broker_df[["date", "net_lots"]], on="date", how="left").fillna(0)
        merged["broker_ratio"] = merged["net_lots"] / (merged["Trading_Volume"] / 1000.0)

        # === Z-score ===
        merged["zscore"] = merged["net_lots"].rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        # === 交易邏輯 ===
        merged["signal"] = 0.0
        position_size = 0.0
        avg_entry_price = None
        max_entry_price = None
        peak_price = None
        first_take_profit_done = False

        for i in range(len(merged)):
            row = merged.iloc[i]
            price_now = row["close"]

            # === 出場條件 ===
            if position_size > 0:
                # Hard Stop (用 max_entry_price 當基準)
                if max_entry_price and price_now <= max_entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    avg_entry_price, max_entry_price, peak_price = None, None, None
                    first_take_profit_done = False
                    continue

                # 第一檔停利
                if (not first_take_profit_done) and avg_entry_price and price_now >= avg_entry_price * (1 + self.first_take_profit):
                    position_size = position_size / 2
                    merged.loc[i, "signal"] = position_size
                    first_take_profit_done = True
                    continue

                # 移動停利
                peak_price = max(peak_price, price_now) if peak_price else price_now
                if peak_price and price_now <= peak_price * (1 - self.trailing_stop):
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    avg_entry_price, max_entry_price, peak_price = None, None, None
                    first_take_profit_done = False
                    continue

                # 籌碼反轉 (連三日賣超)
                if i >= 3 and (merged["net_lots"].iloc[i-2:i+1] < 0).all():
                    merged.loc[i, "signal"] = -1.0
                    position_size = 0.0
                    avg_entry_price, max_entry_price, peak_price = None, None, None
                    first_take_profit_done = False
                    continue

            # === 進場 / 加倉判斷 ===
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)
                zscore_score = min(1.0, row["zscore"] / 5.0)
                strength = 0.6 * ratio_score + 0.4 * zscore_score
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉，不減倉
                    # 更新加權成本
                    if avg_entry_price is None:
                        avg_entry_price = row["close"]
                        max_entry_price = row["close"]
                    else:
                        total_value = avg_entry_price * position_size + row["close"] * (new_position - position_size)
                        avg_entry_price = total_value / new_position
                        # Hard stop 基準：最高進場價
                        max_entry_price = max(max_entry_price, row["close"])

                    position_size = new_position
                    peak_price = row["close"] if peak_price is None else max(peak_price, row["close"])

                merged.loc[i, "signal"] = position_size
            else:
                merged.loc[i, "signal"] = position_size

        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        return merged



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


class BrokerFollowStrategyV12(Strategy):
    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]
    ratio_th: float = 0.05
    zscore_th: float = 2.0
    lookback: int = 60

    stop_loss: float = 0.10       # 強制停損 10%
    trailing_stop: float = 0.05   # 移動停利 5%
    take_profit: float = 0.08     # 停利 8%

    # 🔥 新增爆量買超條件
    volume_surge_lookback: int = 5    # 爆量判斷的觀察天數
    volume_surge_mult: float = 3.0    # 幾倍算爆量

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
                # 強制停損
                if avg_entry_price and price_now <= avg_entry_price * (1 - self.stop_loss):
                    merged.loc[i, "signal"] = -position_size
                    position_size = 0.0
                    avg_entry_price = max_entry_price = peak_price = None
                    continue

                # 部分停利
                if avg_entry_price and price_now >= avg_entry_price * (1 + self.take_profit):
                    sell_lots = max(1.0, position_size / 2)
                    position_size -= sell_lots
                    merged.loc[i, "signal"] = -sell_lots
                    if position_size == 0:
                        avg_entry_price = max_entry_price = peak_price = None
                    continue

                # 移動停利
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
            buy_signal = False

            # 條件 1: broker ratio + zscore
            if row["broker_ratio"] > self.ratio_th and row["zscore"] > self.zscore_th:
                buy_signal = True

            # 條件 2: 爆量買超
            if i >= self.volume_surge_lookback:
                recent_avg = merged["net_lots"].iloc[i-self.volume_surge_lookback:i].mean()
                if recent_avg > 0 and row["net_lots"] > recent_avg * self.volume_surge_mult:
                    buy_signal = True

            if buy_signal:
                # 計算倉位
                ratio_score = min(1.0, row["broker_ratio"] / 0.2)
                zscore_score = min(1.0, row["zscore"] / 5.0) if not pd.isna(row["zscore"]) else 0
                strength = 0.6 * ratio_score + 0.4 * zscore_score
                new_position = round(0.5 + 4.5 * strength, 1)

                if new_position > position_size:  # 只加倉
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
