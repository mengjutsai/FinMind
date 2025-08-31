import pandas as pd
from FinMind.strategies.base_sql import Strategy

class PriceAbove1000(Strategy):
    def create_trade_sign(self, stock_price: pd.DataFrame, additional_dataset_obj=None):
        """
        策略：
        - 當股價第一次突破 1000 時買進 (signal=1)
        - 之後一路持有，不再賣出
        """
        stock_price = stock_price.copy()

        # 檢查欄位名稱
        print("欄位名稱:", stock_price.columns.tolist())

        # 預設 signal = 0
        stock_price["signal"] = 0  

        # 確認 close 欄位是否存在
        if "close" not in stock_price.columns:
            raise ValueError("找不到欄位 'close'，請確認 stock_price 的資料格式")

        # 找出第一次收盤價 > 1000 的 index
        buy_index = stock_price[stock_price["close"] > 1000].index.min()

        if pd.notna(buy_index):  
            buy_date = stock_price.loc[buy_index, "date"]
            buy_price = stock_price.loc[buy_index, "close"]
            print(f"突破點: {buy_date} 收盤 {buy_price} → 進場買進")
            stock_price.loc[buy_index:, "signal"] = 1   # 從突破開始一路持有
        else:
            print("整個期間內，股價都沒破 1000 → 不進場")

        return stock_price
