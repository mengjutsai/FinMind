import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
# ta 0.5.25 的 ATR 參數是 n
try:
    from ta.volatility import AverageTrueRange
    HAS_ATR = True
except Exception:
    HAS_ATR = False

from FinMind.strategies.base_sql import Strategy


class KdTrendRisk(Strategy):
    """
    ta==0.5.25 版相容
    規則（僅做多）：
      入場： KD 黃金交叉 AND 收盤 > MA60 AND MA20 > MA60 AND K < k_entry_max
      出場： KD 死亡交叉 OR 收盤 < MA60 OR 收盤 <= 入場價*(1 - stop_loss_pct)
    參數：
      k_days: KD 參數 (對應 ta 0.5.25 的 n，預設 9)
      fast_ma: 快均天數 (預設 20)
      slow_ma: 慢均天數 (預設 60)
      k_entry_max: 入場 K 值上限，避免過度超買追價 (預設 80)
      stop_loss_pct: 進場價百分比停損 (預設 0.06=6%)
      use_atr_trail: 是否計算 ATR 供之後擴充移動停損（當前策略未用，僅紀錄）
      atr_n: ATR 期間 (預設 14) — 僅當 use_atr_trail=True 且 ta 有 ATR 時計算
    輸出：
      stock_price['signal']: 1=買進, -1=賣出, 0=無動作
    注意：
      需要欄位：['close','max','min']（FinMind 價量表預設都有）
    """

    k_days: int = 9
    fast_ma: int = 20
    slow_ma: int = 60
    k_entry_max: float = 80.0
    stop_loss_pct: float = 0.06
    use_atr_trail: bool = False
    atr_n: int = 14

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = stock_price.copy()

        # --- 基本防呆：確保必備欄位存在 ---
        for col in ["close", "max", "min"]:
            if col not in sp.columns:
                raise KeyError(f"Missing column '{col}' required by KdTrendRisk.")

        # --- KD (ta 0.5.25：n / d_n) ---
        stoch = StochasticOscillator(
            high=sp["max"],
            low=sp["min"],
            close=sp["close"],
            n=self.k_days,  # 舊版參數名
            d_n=3
        )
        sp["K"] = stoch.stoch()
        sp["D"] = stoch.stoch_signal()

        # --- 均線 ---
        sp[f"ma{self.fast_ma}"] = sp["close"].rolling(self.fast_ma, min_periods=1).mean()
        sp[f"ma{self.slow_ma}"] = sp["close"].rolling(self.slow_ma, min_periods=1).mean()

        # --- KD 交叉（前一日 K<D，當日 K>=D 為黃金交叉；相反為死亡交叉） ---
        sp["golden"] = (sp["K"].shift(1) < sp["D"].shift(1)) & (sp["K"] >= sp["D"])
        sp["death"]  = (sp["K"].shift(1) > sp["D"].shift(1)) & (sp["K"] <= sp["D"])

        # --- （可選）ATR 計算以供將來拓展移動停損 ---
        if self.use_atr_trail and HAS_ATR:
            atr = AverageTrueRange(
                high=sp["max"], low=sp["min"], close=sp["close"], n=self.atr_n, fillna=False
            )
            sp["ATR"] = atr.average_true_range()
        else:
            sp["ATR"] = np.nan

        # --- 產生交易訊號（含部位狀態）---
        signal = np.zeros(len(sp), dtype=int)
        holding = False
        entry_price = np.nan

        ma_fast_col = f"ma{self.fast_ma}"
        ma_slow_col = f"ma{self.slow_ma}"

        for i in range(len(sp)):
            c = float(sp.iat[i, sp.columns.get_loc("close")])
            k = float(sp.iat[i, sp.columns.get_loc("K")]) if not np.isnan(sp.iat[i, sp.columns.get_loc("K")]) else np.nan
            ma_fast = sp.iat[i, sp.columns.get_loc(ma_fast_col)]
            ma_slow = sp.iat[i, sp.columns.get_loc(ma_slow_col)]
            is_golden = bool(sp.iat[i, sp.columns.get_loc("golden")])
            is_death  = bool(sp.iat[i, sp.columns.get_loc("death")])

            # 入場（只在多頭結構 & K 未過熱）
            if not holding:
                if (
                    is_golden
                    and pd.notna(ma_fast) and pd.notna(ma_slow)
                    and c > ma_slow
                    and ma_fast > ma_slow
                    and pd.notna(k) and k < self.k_entry_max
                ):
                    signal[i] = 1
                    holding = True
                    entry_price = c
            else:
                # 出場（任一條件觸發）
                stop_loss_hit = (c <= entry_price * (1 - self.stop_loss_pct)) if pd.notna(entry_price) else False
                trend_break   = (pd.notna(ma_slow) and c < ma_slow)
                if is_death or stop_loss_hit or trend_break:
                    signal[i] = -1
                    holding = False
                    entry_price = np.nan

        sp["signal"] = signal

        # 清理暫存欄（保留 K/D/均線/ATR 方便你後續分析就不刪了；若要精簡可改這行）
        sp.drop(columns=["golden", "death"], inplace=True)

        return sp
