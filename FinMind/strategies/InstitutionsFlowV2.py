import typing
import numpy as np
import pandas as pd

from FinMind.schema.data import Dataset
from FinMind.strategies.base import Strategy


class InstitutionsFlowV2(Strategy):
    """
    三大法人「相對規模」+ 趨勢/量能濾網 + 風控 + 持倉狀態
    入場（僅多單）：
      - 三大法人 n 日累積淨買超的 z-score > z_th，且 近1日淨買超佔20日成交量比例 > flow_ratio_th
      - 多頭結構：close > MA60 且 MA20 > MA60
      - 價格尚未過度噴出：close <= 20日高點 * (1 + no_surge_pct)
      - 量能放大：今日成交量 > 20日均量 * vol_ratio_th
    出場：
      - 三大法人轉為顯著淨賣出（z-score < -z_exit）
      - 或 跌破 MA60
      - 或 停損（相對進場價）

    參數：
      n_days:         計算累積法人淨買超的視窗（預設 5）
      z_th:           入場 z-score 門檻（預設 1.0）
      z_exit:         出場 z-score 門檻（預設 0.5）
      flow_ratio_th:  近1日淨買超 / 20日成交量 比例門檻（預設 0.10 = 10%）
      no_surge_pct:   允許距 20 日高點的噴出幅度（預設 0.03 = 3%）
      vol_ratio_th:   量能放大倍數（相對 20 日均量，預設 1.3）
      stop_loss_pct:  停損（預設 0.06 = 6%）
      use_next_day:   是否假設隔日才能執行（預設 True）

    注意：
      - 需要 stock_price 內含：['date','stock_id','close','Trading_Volume']（FinMind 命名常為 Trading_Volume）
      - 若你的欄位不同，請在程式內對應修改。
    """

    n_days: int = 5
    z_th: float = 1.0
    z_exit: float = 0.5
    flow_ratio_th: float = 0.10
    no_surge_pct: float = 0.03
    vol_ratio_th: float = 1.3
    stop_loss_pct: float = 0.06
    use_next_day: bool = True

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.sort_values("date").copy()
        sp.index = range(len(sp))

        required_cols = ["date", "stock_id", "close"]
        for col in required_cols:
            if col not in sp.columns:
                raise KeyError(f"Missing column '{col}' required by InstitutionsFlowV2.")

        # 成交量欄位名稱可能是 'Trading_Volume' 或 'TradingVolume'，做個容錯
        vol_col = "Trading_Volume" if "Trading_Volume" in sp.columns else (
            "TradingVolume" if "TradingVolume" in sp.columns else None
        )
        if vol_col is None:
            raise KeyError("Missing volume column ('Trading_Volume' or 'TradingVolume').")
        sp["vol"] = sp[vol_col].astype(float)

        # 取三大法人資料並彙總（買-賣）
        inst = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockInstitutionalInvestorsBuySell,
            data_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        inst = (
            inst.groupby(["date", "stock_id"], as_index=False)
            .agg({"buy": np.sum, "sell": np.sum})
        )
        inst["net"] = inst["buy"] - inst["sell"]

        sp = pd.merge(
            sp, inst[["date", "stock_id", "net"]],
            on=["date", "stock_id"], how="left"
        ).fillna({"net": 0.0})

        # 均線與價量統計
        sp["ma20"] = sp["close"].rolling(20, min_periods=1).mean()
        sp["ma60"] = sp["close"].rolling(60, min_periods=1).mean()
        sp["vol_ma20"] = sp["vol"].rolling(20, min_periods=1).mean()
        sp["hhv20"] = sp["close"].rolling(20, min_periods=1).max()

        # n日累積法人淨買超 + z-score
        sp["net_n"] = sp["net"].rolling(self.n_days, min_periods=1).sum()
        mu = sp["net_n"].rolling(20, min_periods=5).mean()
        sd = sp["net_n"].rolling(20, min_periods=5).std(ddof=0)
        sp["z"] = (sp["net_n"] - mu) / (sd.replace(0, np.nan))

        # 規模正規化（近1日淨買超 / 20日成交量）
        sp["flow_ratio"] = sp["net"] / (sp["vol_ma20"].replace(0, np.nan))

        # 入場條件
        sp["long_setup"] = (
            (sp["z"] > self.z_th)
            & (sp["flow_ratio"] > self.flow_ratio_th)
            & (sp["close"] > sp["ma60"])
            & (sp["ma20"] > sp["ma60"])
            & (sp["close"] <= sp["hhv20"] * (1 + self.no_surge_pct))
            & (sp["vol"] > sp["vol_ma20"] * self.vol_ratio_th)
        )

        # 出場條件
        sp["exit_setup"] = (
            (sp["z"] < -self.z_exit)
            | (sp["close"] < sp["ma60"])
        )

        # 產生持倉式訊號
        signal = np.zeros(len(sp), dtype=int)
        holding = False
        entry_price = np.nan

        for i in range(len(sp)):
            c = float(sp.at[i, "close"])
            if not holding:
                if bool(sp.at[i, "long_setup"]):
                    signal[i] = 1
                    holding = True
                    entry_price = c
            else:
                stop_loss_hit = c <= entry_price * (1 - self.stop_loss_pct)
                if bool(sp.at[i, "exit_setup"]) or stop_loss_hit:
                    signal[i] = -1
                    holding = False
                    entry_price = np.nan

        sp["signal"] = signal

        # 可選：如果你要嚴格避免前視，強制所有動作隔日執行
        if self.use_next_day:
            sp["signal"] = sp["signal"].shift(1).fillna(0).astype(int)

        # 清理暫存欄位（保留一些診斷欄位以便檢視）
        # sp = sp.drop(columns=["long_setup", "exit_setup"])  # 若要精簡可打開

        return sp
