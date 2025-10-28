import pandas as pd
from FinMind.strategies.base_sql import Strategy
from FinMind.indicators import add_continue_holding_indicators


# -----------------------------
# 週期策略
# -----------------------------
class ContinueHolding7(Strategy):
    """每 7 天買進一次"""
    buy_freq_day = 7
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = add_continue_holding_indicators(stock_price, buy_freq_day=self.buy_freq_day)
        stock_price["signal"] = stock_price["DollarCostAveraging"]
        return stock_price

class ContinueHolding15(Strategy):
    """每 15 天買進一次"""
    buy_freq_day = 15
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = add_continue_holding_indicators(stock_price, buy_freq_day=self.buy_freq_day)
        stock_price["signal"] = stock_price["DollarCostAveraging"]
        return stock_price

class ContinueHolding30(Strategy):
    """每 30 天買進一次"""
    buy_freq_day = 30
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        stock_price = add_continue_holding_indicators(stock_price, buy_freq_day=self.buy_freq_day)
        stock_price["signal"] = stock_price["DollarCostAveraging"]
        return stock_price


# -----------------------------
# 固定單日策略 Day1 ~ Day28
# -----------------------------
class ContinueHoldingDay1(Strategy):
    """每月 1 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 1, "signal"] = 1
        return df

class ContinueHoldingDay2(Strategy):
    """每月 2 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 2, "signal"] = 1
        return df

class ContinueHoldingDay3(Strategy):
    """每月 3 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 3, "signal"] = 1
        return df

class ContinueHoldingDay4(Strategy):
    """每月 4 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 4, "signal"] = 1
        return df

class ContinueHoldingDay5(Strategy):
    """每月 5 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 5, "signal"] = 1
        return df

class ContinueHoldingDay6(Strategy):
    """每月 6 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 6, "signal"] = 1
        return df

class ContinueHoldingDay7(Strategy):
    """每月 7 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 7, "signal"] = 1
        return df

class ContinueHoldingDay8(Strategy):
    """每月 8 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 8, "signal"] = 1
        return df

class ContinueHoldingDay9(Strategy):
    """每月 9 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 9, "signal"] = 1
        return df

class ContinueHoldingDay10(Strategy):
    """每月 10 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 10, "signal"] = 1
        return df

class ContinueHoldingDay11(Strategy):
    """每月 11 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 11, "signal"] = 1
        return df

class ContinueHoldingDay12(Strategy):
    """每月 12 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 12, "signal"] = 1
        return df

class ContinueHoldingDay13(Strategy):
    """每月 13 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 13, "signal"] = 1
        return df

class ContinueHoldingDay14(Strategy):
    """每月 14 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 14, "signal"] = 1
        return df

class ContinueHoldingDay15(Strategy):
    """每月 15 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 15, "signal"] = 1
        return df

class ContinueHoldingDay16(Strategy):
    """每月 16 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 16, "signal"] = 1
        return df

class ContinueHoldingDay17(Strategy):
    """每月 17 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 17, "signal"] = 1
        return df

class ContinueHoldingDay18(Strategy):
    """每月 18 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 18, "signal"] = 1
        return df

class ContinueHoldingDay19(Strategy):
    """每月 19 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 19, "signal"] = 1
        return df

class ContinueHoldingDay20(Strategy):
    """每月 20 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 20, "signal"] = 1
        return df

class ContinueHoldingDay21(Strategy):
    """每月 21 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 21, "signal"] = 1
        return df

class ContinueHoldingDay22(Strategy):
    """每月 22 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 22, "signal"] = 1
        return df

class ContinueHoldingDay23(Strategy):
    """每月 23 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 23, "signal"] = 1
        return df

class ContinueHoldingDay24(Strategy):
    """每月 24 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 24, "signal"] = 1
        return df

class ContinueHoldingDay25(Strategy):
    """每月 25 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 25, "signal"] = 1
        return df

class ContinueHoldingDay26(Strategy):
    """每月 26 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 26, "signal"] = 1
        return df

class ContinueHoldingDay27(Strategy):
    """每月 27 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 27, "signal"] = 1
        return df

class ContinueHoldingDay28(Strategy):
    """每月 28 號買進"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"] == 28, "signal"] = 1
        return df


# -----------------------------
# 多日策略
# -----------------------------
class ContinueHoldingDay1_15(Strategy):
    """每月 1 號與 15 號"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"].isin([1, 15]), "signal"] = 0.5
        return df

class ContinueHoldingDay5_20(Strategy):
    """每月 5 號與 20 號"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"].isin([5, 20]), "signal"] = 0.5
        return df

class ContinueHoldingDay10_25(Strategy):
    """每月 10 號與 25 號"""
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = stock_price.copy(); df["signal"] = 0
        df["day"] = pd.to_datetime(df["date"]).dt.day
        df.loc[df["day"].isin([10, 25]), "signal"] = 0.5
        return df


