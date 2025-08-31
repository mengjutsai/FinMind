import typing
import numpy as np
import pandas as pd

from FinMind.schema.data import Dataset
from FinMind.strategies.base_sql import Strategy


class InstitutionsFlowV3(Strategy):
    """
    三大法人「相對規模」+ 趨勢/量能濾網 + 信心度下單
    入場：
      - 符合多頭結構與濾網
      - 部位大小依 z-score 與 flow_ratio 動態調整
    出場：
      - z-score 顯著轉負、跌破均線或停損
      - 出場張數依負向強度動態調整

    參數：
      n_days:         計算累積法人淨買超的視窗（預設 5）
      z_th:           入場最低 z-score 門檻
      flow_ratio_th:  入場最低 flow 比例
      no_surge_pct:   噴出過熱過濾
      vol_ratio_th:   量能濾網
      stop_loss_pct:  停損
      use_next_day:   是否隔日才能執行
      size_scale:     部位放大因子（控制 signal 數值大小）
    """

    n_days: int = 5
    z_th: float = 1.0
    flow_ratio_th: float = 0.1
    no_surge_pct: float = 0.03
    vol_ratio_th: float = 1.3
    stop_loss_pct: float = 0.06
    use_next_day: bool = True
    size_scale: float = 100  # 每單位強度轉換成多少張

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.sort_values("date").copy()
        sp.index = range(len(sp))

        vol_col = "Trading_Volume" if "Trading_Volume" in sp.columns else (
            "TradingVolume" if "TradingVolume" in sp.columns else None
        )
        if vol_col is None:
            raise KeyError("Missing volume column ('Trading_Volume' or 'TradingVolume').")
        sp["vol"] = sp[vol_col].astype(float)

        # 取三大法人資料並彙總
        inst = self.data_loader.get_data(
            dataset=Dataset.TaiwanStockInstitutionalInvestorsBuySell,
            data_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        inst = inst.groupby(["date", "stock_id"], as_index=False).agg(
            {"buy": np.sum, "sell": np.sum}
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

        # 規模正規化
        sp["flow_ratio"] = sp["net"] / (sp["vol_ma20"].replace(0, np.nan))

        # 入場 setup
        sp["long_setup"] = (
            (sp["z"] > self.z_th)
            & (sp["flow_ratio"] > self.flow_ratio_th)
            & (sp["close"] > sp["ma60"])
            & (sp["ma20"] > sp["ma60"])
            & (sp["close"] <= sp["hhv20"] * (1 + self.no_surge_pct))
            & (sp["vol"] > sp["vol_ma20"] * self.vol_ratio_th)
        )

        # 信心度（入場張數）
        sp["long_size"] = (
            np.maximum(0, sp["z"]) * sp["flow_ratio"] * self.size_scale
        )

        # 出場 setup
        sp["exit_setup"] = (
            (sp["z"] < -0.5)  # 可以獨立設 exit 門檻
            | (sp["close"] < sp["ma60"])
        )
        sp["exit_size"] = (
            np.maximum(0, -sp["z"]) * self.size_scale
        )

        # 持倉訊號：用強度控制倉位
        signal = np.zeros(len(sp))
        holding = False
        entry_price = np.nan

        for i in range(len(sp)):
            c = float(sp.at[i, "close"])
            if not holding:
                if bool(sp.at[i, "long_setup"]):
                    signal[i] = sp.at[i, "long_size"]
                    holding = True
                    entry_price = c
            else:
                stop_loss_hit = c <= entry_price * (1 - self.stop_loss_pct)
                if bool(sp.at[i, "exit_setup"]) or stop_loss_hit:
                    signal[i] = -sp.at[i, "exit_size"]
                    holding = False
                    entry_price = np.nan

        sp["signal"] = signal

        # 延遲一天執行
        if self.use_next_day:
            sp["signal"] = sp["signal"].shift(1).fillna(0)

        return sp
