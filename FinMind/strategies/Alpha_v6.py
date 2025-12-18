############################################################
#                    Alpha_v6                        #
#          Pattern Breakout (YOLO) × Broker Factor        #
############################################################
import os
import subprocess
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from FinMind.strategies.base_sql import Strategy
pd.set_option('display.max_columns', None)
from typing import Union



def _plot_worker(task):
    try:
        subprocess.run(task["cmd"], check=True)
        return True
    except Exception:
        return False


############################################################
#                 Broker Factor Engine                    #
############################################################
class FactorEngine:

    def __init__(
        self,
        enable_pattern=True,
        enable_broker=True,
        enable_volume=True,
        enable_volatility=True,

        weight_pattern=1.0,
        weight_broker=1.0,
        weight_volume=0.8,       # ← 你原本的 weight
        weight_vol_break=1.0,
        weight_volume_short=0.8,
        buy_threshold=0.5,
        max_add=3
    ):
        self.enable_pattern = enable_pattern
        self.enable_broker = enable_broker
        self.enable_volume = enable_volume
        self.enable_volatility = enable_volatility

        self.w_pattern = weight_pattern
        self.w_broker = weight_broker
        self.w_volume = weight_volume
        self.w_vol_break = weight_vol_break
        self.w_volume_short = weight_volume_short

        self.buy_threshold = buy_threshold
        self.max_add = max_add


        # Pattern class weights（你原本的保持）
        self.pattern_class_weight = {
            "Up-Triangle": 1.0,
            "Up-W": 1.2,
            "Up-Head-Shoulder-Bottom": 0.9,
            "Up-Breakout": 0.0,

            "Down-Triangle": 0.0,
            "Down-M": 0.0,
            "Down-Head-Shoulder-Top": 0.0,
            "Down-Breakout": 0.0,
        }


    # ==========================================================
    # Price Trend Factor（5/10/20 趨勢強度）
    # ==========================================================
    def price_trend_factors(self, sp, idx, windows=[5, 10, 20]):
        if idx < max(windows):
            return 0.0, 0.0

        scores = []
        for w in windows:
            y = sp["close"].iloc[idx-w+1:idx+1].values
            x = np.arange(w)
            slope = np.polyfit(x, y, 1)[0]
            slope_norm = slope / (np.mean(y) + 1e-9)
            scores.append(np.tanh(slope_norm * 5))

        trend = float(np.mean(scores))

        long_factor = max(0, trend)
        short_factor = max(0, -trend)

        return long_factor, short_factor

    # ==========================================================
    # Pullback Quality Factor（回檔品質）
    # ==========================================================
    def pullback_factors(self, sp, idx):
        if idx < 3:
            return 0.0, 0.0

        close = sp["close"]
        vol = sp["Trading_Volume"]
        high = sp["max"]
        low = sp["min"]

        # 三天變化
        drop = (close.iloc[idx] - close.iloc[idx-3]) / (close.iloc[idx-3] + 1e-9)

        # 陰線／陽線影線
        lower_shadow = close.iloc[idx] - low.iloc[idx]
        upper_shadow = high.iloc[idx] - close.iloc[idx]

        shadow_score = np.tanh((lower_shadow - upper_shadow) /
                            (high.iloc[idx] - low.iloc[idx] + 1e-9))

        # 量縮
        vol_ratio = vol.iloc[idx] / (vol.iloc[idx-3:idx].mean() + 1e-9)
        vol_score = np.tanh((1 - vol_ratio) * 2)

        # 跌幅小 → 多
        drop_score = np.tanh(-drop * 5)

        # 多頭版本（健康回檔）
        long_raw = 0.5 * vol_score + 0.3 * shadow_score + 0.2 * drop_score
        long_factor = float(np.tanh(long_raw))

        # 空頭版本（反彈弱）
        rebound = -drop  # 三日漲跌反向
        weak_rebound = np.tanh(-shadow_score)  # 上影線長
        weak_vol = np.tanh((vol_ratio - 1) * 2)  # 反彈放量不過高

        short_raw = 0.4 * weak_rebound + 0.4 * weak_vol + 0.2 * rebound
        short_factor = float(np.tanh(short_raw))

        return long_factor, short_factor


    # ==========================================================
    # Breakout Quality Factor（突破 K 線品質）
    # ==========================================================
    def breakout_breakdown_factors(self, sp, idx, window=20):
        if idx < window:
            return 0.0, 0.0

        close = sp["close"]
        vol = sp["Trading_Volume"]
        high = sp["max"]
        low = sp["min"]

        prev_high = high.iloc[idx-window:idx].max()
        prev_low = low.iloc[idx-window:idx].min()

        # ============================
        # 多頭：向上突破品質
        # ============================
        is_break = close.iloc[idx] > prev_high

        if is_break:
            vol_mean = vol.iloc[idx-window:idx].mean()
            vol_score = np.tanh((vol.iloc[idx] / (vol_mean + 1e-9) - 1))
            body = close.iloc[idx] - sp["open"].iloc[idx]
            body_score = np.tanh(body / (close.iloc[idx] + 1e-9) * 5)

            if idx < len(close) - 1:
                ft = (close.iloc[idx+1] - close.iloc[idx]) / close.iloc[idx]
                follow_score = np.tanh(ft * 10)
            else:
                follow_score = 0

            long_raw = 0.45 * vol_score + 0.35 * body_score + 0.20 * follow_score
            long_factor = float(np.tanh(long_raw))
        else:
            long_factor = 0.0

        # ============================
        # 空頭：向下跌破品質
        # ============================
        is_breakdown = close.iloc[idx] < prev_low

        if is_breakdown:
            vol_mean = vol.iloc[idx-window:idx].mean()
            vol_score = np.tanh((vol.iloc[idx] / (vol_mean + 1e-9) - 1))
            body = sp["open"].iloc[idx] - close.iloc[idx]
            body_score = np.tanh(body / (close.iloc[idx] + 1e-9) * 5)

            if idx < len(close) - 1:
                ft = (close.iloc[idx+1] - close.iloc[idx]) / close.iloc[idx]
                follow_score = np.tanh(-ft * 10)  # 跌後再跌 → 空頭更強
            else:
                follow_score = 0

            short_raw = 0.45 * vol_score + 0.35 * body_score + 0.20 * follow_score
            short_factor = float(np.tanh(short_raw))
        else:
            short_factor = 0.0

        return long_factor, short_factor


    # ==========================================================
    # Price Box Factor（箱體壓縮品質）
    # ==========================================================
    def price_box_factors(self, sp, idx, window=20):
        if idx < window:
            return 0.0, 0.0

        high = sp["max"].iloc[idx-window:idx]
        low = sp["min"].iloc[idx-window:idx]
        close = sp["close"].iloc[idx]

        box_range = (high.max() - low.min()) / (close + 1e-9)
        compression_score = np.tanh((0.2 - box_range) * 5)

        pos = (close - low.min()) / (high.max() - low.min() + 1e-9)
        pos_score = np.tanh((pos - 0.5) * 3)

        # 多：箱體上緣壓縮
        long_factor = float(0.7 * compression_score + 0.3 * pos_score)

        # 空：箱體下緣壓縮
        short_factor = float(0.7 * compression_score + 0.3 * (-pos_score))

        return long_factor, short_factor


    # ==========================================================
    # Relative Strength Factor（相對強弱）
    # ==========================================================
    def relative_strength_factors(self, sp, idx, index_close, win=20):
        if idx < win:
            return 0.0, 0.0

        stock_ret = sp["close"].pct_change(win).iloc[idx]
        index_ret = index_close.pct_change(win).iloc[idx]
        rs = stock_ret - index_ret

        long_factor = float(np.tanh(rs * 5))      # 強於大盤
        short_factor = float(np.tanh(-rs * 5))    # 弱於大盤

        return long_factor, short_factor

    # ==========================================================
    # Price Long/Short Factor（整合五大價因子）
    # ==========================================================
    def combined_price_factors(
        self,
        trend_long, trend_short,
        pull_long, pull_short,
        brk_long, brk_short,
        box_long, box_short,
        rs_long, rs_short
    ):
        # 多頭：五者平均（可加權）
        price_long = np.tanh(
            0.25 * trend_long +
            0.20 * pull_long +
            0.25 * brk_long +
            0.15 * box_long +
            0.15 * rs_long
        )

        # 空頭：五者平均（可加權）
        price_short = np.tanh(
            0.25 * trend_short +
            0.20 * pull_short +
            0.25 * brk_short +
            0.15 * box_short +
            0.15 * rs_short
        )

        return float(price_long), float(price_short)


    def normalize_price_factors(self, f_price_long, f_price_short):
        raw = f_price_long - f_price_short     # -1 ~ +1
        f_price = np.tanh(raw)                 # 再次壓縮，避免噪音
        f_price_long_final  = max(f_price, 0)
        f_price_short_final = max(-f_price, 0)
        return f_price_long_final, f_price_short_final, f_price


    # ==========================================================
    # Volume Compression（原本保留）
    # ==========================================================
    def volume_compression_factor(self, vol_ema_fast, vol_ema_slow):
        if not self.enable_volume:
            return 0
        if vol_ema_slow <= 0:
            return 0
        r = vol_ema_fast / vol_ema_slow
        score = np.tanh((1 - r) * 2)
        return max(0, score)

    # ==========================================================
    # Volatility Compression（原本保留）
    # ==========================================================
    def volatility_factor(self, vola_ema_fast, vola_ema_slow):
        if not self.enable_volatility:
            return 0
        if vola_ema_slow <= 0:
            return 0
        r = vola_ema_fast / vola_ema_slow
        score = np.tanh((1 - r) * 2)
        return max(0, score)

    # ==========================================================
    # Volume Breakout（原本保留）
    # ==========================================================
    def volume_breakout_factor(self, vol_today, vol_slow_ema):
        if vol_slow_ema <= 0:
            return 0
        r = vol_today / vol_slow_ema
        score = np.tanh((r - 1.0))
        return max(0, score)

    def _calc_vol_breakout(self, sp, idx, slow=20):
        if idx < slow:
            return 0, 0
        today = sp["Trading_Volume"].iloc[idx]
        slow_ema = sp["Trading_Volume"].ewm(span=slow, adjust=False).mean().iloc[idx]
        return today, slow_ema

    # ==========================================================
    # Volume Dry-up（新增）
    # ==========================================================
    def volume_dryup_factor(self, sp, idx, window=20):
        if idx < window:
            return 0
        v_today = sp["Trading_Volume"].iloc[idx]
        v_mean = sp["Trading_Volume"].iloc[idx-window:idx].mean()

        # mean / today 越大 → 地板量越乾淨
        dry = np.tanh((v_mean / (v_today + 1e-9)) - 1)
        return max(0, dry)

    # ==========================================================
    # Volume Follow-through（新增）
    # ==========================================================
    def volume_follow_factor(self, sp, idx):
        if idx < 1:
            return 0

        vol_today = sp["Trading_Volume"].iloc[idx]
        vol_prev = sp["Trading_Volume"].iloc[idx-1]
        close_today = sp["close"].iloc[idx]
        close_prev = sp["close"].iloc[idx-1]

        if close_today > close_prev and vol_today > vol_prev:
            return np.tanh((vol_today / (vol_prev + 1e-9)) - 1)

        return 0

    # ==========================================================
    # Volume–Price Divergence（新增）
    # ==========================================================
    def volume_price_divergence(self, sp, idx, window=10):
        if idx < window:
            return 0

        price = sp["close"].iloc[idx-window:idx].values
        vol = sp["Trading_Volume"].iloc[idx-window:idx].values

        price_trend = np.polyfit(np.arange(window), price, 1)[0]
        vol_trend = np.polyfit(np.arange(window), vol, 1)[0]

        # 主力吸貨：量上升，價格平/升
        if vol_trend > 0 and price_trend >= 0:
            return np.tanh((vol_trend / (abs(price_trend) + 1e-9)) * 0.001)

        # 主力派貨：量上升，價格下降
        if vol_trend > 0 and price_trend < 0:
            return -np.tanh((vol_trend / (abs(price_trend) + 1e-9)) * 0.001)

        return 0

    # ==========================================================
    # **最終 Volume 因子（整合版）**
    # ==========================================================
    def volume_long_factor(self, sp, idx, vol_fast, vol_slow, vol_today, slow_ema):
        f_compress = self.volume_compression_factor(vol_fast, vol_slow)
        f_break = self.volume_breakout_factor(vol_today, slow_ema)
        f_dry = self.volume_dryup_factor(sp, idx)
        f_follow = self.volume_follow_factor(sp, idx)
        f_div = self.volume_price_divergence(sp, idx)

        raw = (
            0.35 * f_compress +
            0.25 * f_dry +
            0.25 * f_break +
            0.15 * f_follow +
            0.10 * f_div        # 以前是裸加，改成*0.1
        )

        return np.tanh(raw)


    # ==========================================================
    # Short Volume Factor（空頭量價警訊）
    # ==========================================================
    def volume_short_factor(self, sp, idx):
        """
        = 空頭風險因子（越高代表越危險）
        包含：
        - 爆量長黑
        - 價跌量增（派貨）
        - 高檔背離
        - 量增價跌 divergence
        """
        if idx < 3:
            return 0.0

        vol_today = sp["Trading_Volume"].iloc[idx]
        vol_prev  = sp["Trading_Volume"].iloc[idx-1]
        close_today = sp["close"].iloc[idx]
        close_prev  = sp["close"].iloc[idx-1]
        high = sp["max"].iloc[idx]
        low  = sp["min"].iloc[idx]        

        score = 0.0

        # 1. 價跌量增（派貨訊號）
        if close_today < close_prev and vol_today > vol_prev:
            s = np.tanh((vol_today / (vol_prev + 1e-9)) - 1)
            score += 0.40 * max(0, s)

        # 2. 爆量長黑（主力倒貨）
        body = abs(close_today - close_prev)
        if (close_today < close_prev) and (vol_today > vol_prev * 1.3) and (body > (high - low) * 0.5):
            score += 0.40

        # 3. 量價背離（量增價跌）
        if idx > 10:
            price = sp["close"].iloc[idx-10:idx].values
            vol   = sp["Trading_Volume"].iloc[idx-10:idx].values
            p_trend = np.polyfit(np.arange(10), price, 1)[0]
            v_trend = np.polyfit(np.arange(10), vol, 1)[0]
            if v_trend > 0 and p_trend < 0:
                s = np.tanh((v_trend / (abs(p_trend) + 1)) * 0.001)
                score += 0.20 * max(0, s)

        return min(score, 1.0)

    def normalize_volume_factors(self, f_vol_long, f_vol_short):
        raw = f_vol_long - f_vol_short        # -1 ~ +1
        f_vol = np.tanh(raw)                  # 再 normalize
        f_vol_long_final  = max(f_vol, 0)
        f_vol_short_final = max(-f_vol, 0)
        return f_vol_long_final, f_vol_short_final, f_vol


    # ==========================================================
    # Vola Event Detector（事件型態分類器）
    # ==========================================================
    def vola_event_detector(self, sp, idx, vola_window=20, vol_window=20):
        """
        分類 vola × volume × broker → 事件型態
        回傳：
            event_id: 0~5
            event_score: -1 ~ +1（可直接加到 total_factor）
        """

        # --- 基本保護 ---
        if idx < max(vola_window, vol_window):
            return 0, 0.0

        # -----------------------------
        # 1. 計算 vola（波動率）
        # -----------------------------
        returns = sp["close"].pct_change().iloc[idx-vola_window:idx]
        vola = returns.std()

        vola_mean = returns.std()     # 過去 vola_window
        vola_slow = sp["close"].pct_change().iloc[idx-60:idx].std()

        vola_rel = vola_mean / (vola_slow + 1e-9)   # >1 = 大波動

        # -----------------------------
        # 2. 量能
        # -----------------------------
        vol_today = sp["Trading_Volume"].iloc[idx]
        vol_mean  = sp["Trading_Volume"].iloc[idx-vol_window:idx].mean()

        vol_rel = vol_today / (vol_mean + 1e-9)     # <1 = 小量, >1 = 大量

        # -----------------------------
        # 3. 價格方向
        # -----------------------------
        close_today = sp["close"].iloc[idx]
        close_prev  = sp["close"].iloc[idx-1]

        price_dir = close_today - close_prev      # >0 = 上, <0 = 下

        # -----------------------------
        # 4. Broker 流向
        # 你外部已經算 slope / accel，這裡簡化用 sign
        # -----------------------------
        broker_slope  = sp.get("broker_slope", pd.Series([0]*len(sp))).iloc[idx]
        broker_accel  = sp.get("broker_accel", pd.Series([0]*len(sp))).iloc[idx]

        broker_sig = np.tanh((broker_slope + broker_accel) / 2000)  # -1~+1

        # ======================================================
        # **開始分類事件**
        # ======================================================

        # 條件化門檻（可微調）
        HIGH_VOLA = vola_rel > 1.3
        LOW_VOL   = vol_rel < 0.7
        HIGH_VOL  = vol_rel > 1.3
        BROKER_BUY = broker_sig > 0.2
        BROKER_SELL = broker_sig < -0.2

        # 0. 正常盤
        if not HIGH_VOLA:
            return 0, 0.0

        # 1. Shakeout（洗盤）
        # 大波動、小量、價格下影線 or 微跌、broker 買
        if HIGH_VOLA and LOW_VOL and BROKER_BUY and price_dir <= 0:
            return 1, 0.3     # 小幅正分數 = 再累積訊號

        # 2. Breakout（突破攻擊）
        if HIGH_VOLA and HIGH_VOL and BROKER_BUY and price_dir > 0:
            return 2, 0.8     # 強烈正分數

        # 3. Panic Dump（恐慌倒貨）
        if HIGH_VOLA and HIGH_VOL and BROKER_SELL and price_dir < 0:
            return 3, -0.8    # 強烈負分數

        # 4. Exhaustion（高檔鈍化）
        # 大波動、低量、價格上漲、broker 中性
        if HIGH_VOLA and LOW_VOL and price_dir > 0 and not BROKER_BUY:
            return 4, -0.3    # 高檔上不去

        # 5. Liquidity Crash（流動性蒸發 / 崩盤前兆）
        # 大波動、低量、broker 賣、價格下跌
        if HIGH_VOLA and LOW_VOL and BROKER_SELL and price_dir < 0:
            return 5, -1.0    # 直接視為重空訊號

        # 其他情況：高波動但不構成事件
        return 0, 0.0



    # ==========================================================
    # Pattern Factor
    # ==========================================================
    def pattern_factor(self, label, tf_weight):
        if not self.enable_pattern:
            return 0.0
        base_w = self.pattern_class_weight.get(label, 0.0)
        return base_w * tf_weight



    ############################################################
    #                 Broker Factor (EMA Version)
    ############################################################
    def broker_factor(self, strength, z_buy, slope, accel):
        """
        你在其他地方把買超連續性 streak 改為 slope/accel
        所以這邊全連續化，不需要 streak_days。
        """

        if not self.enable_broker:
            return 0.0

        # Broker modules:

        # (1) 強度（用 tanh 做連續化）
        strength_score = np.tanh(strength / 3.0)

        # (2) Z-score（限制在 0~2 標準差）
        z_score = np.tanh((z_buy or 0) / 3.0)

        # (3) 動能 slope -> 越大越強
        slope_score = np.tanh(slope / 1500)

        # (4) 加速度 accel -> hedge fund 主力因子
        accel_score = np.tanh(accel / 1500)

        return (
            0.40 * accel_score +      # 加速度比 slope 更強，權重最高
            0.25 * slope_score +
            0.20 * strength_score +
            0.15 * z_score
        )

    def vola_confidence(self, vola_fast, vola_slow):
        # vola 沒有方向 → 改成 confidence multiplier
        if vola_slow <= 0:
            return 1.0

        r = vola_fast / (vola_slow + 1e-9)

        # r > 1 → 波動上升 → 趨勢容易延續 → boost > 1
        # r < 1 → 波動壓縮 → 突破失敗機率變高 → boost < 1
        boost = 1 + np.tanh((r - 1) * 1.5) * 0.3

        return float(np.clip(boost, 0.7, 1.3))


    # ==========================================================
    # Total Factor
    # ==========================================================
    def total_factor(
        self,
        f_pattern,          # 多方向
        f_broker,           # 多/空方向
        f_vol_net,          # 多空 volume
        vola_fast, vola_slow,
        f_price_net         # 多空 price
    ):

        # 1) 波動強度影響趨勢與量 → confidence
        vola_boost = self.vola_confidence(vola_fast, vola_slow)

        # 2) 合成原始訊號（raw direction score）
        raw = (
            self.w_pattern   * f_pattern +
            self.w_broker    * f_broker +
            self.w_volume    * (f_vol_net * vola_boost) +
            1.0              * (f_price_net * vola_boost)
        )

        # 3) Normalize to [-1, +1]
        return float(np.tanh(raw))

    def should_buy(self, total_factor):
        return total_factor >= self.buy_threshold

    def add_units(self, total_factor):
        return min(1 + int(total_factor), self.max_add)


############################################################
#                    Alpha_v6 Strategy
############################################################


class Alpha_v6(Strategy):

    # YOLO 模型與腳本
    plot_script = "/Users/meng-jutsai/Stock/FiveB/script/plot_from_sql.py"
    predict_script = "/Users/meng-jutsai/Stock/FiveB/script/predict_seg.py"
    model_path = "/Users/meng-jutsai/Stock/FiveB/runs/segment/BestModel/best.pt"

    # 型態分類
    long_labels = {"Up-Triangle", "Up-W", "Up-Head-Shoulder-Bottom"}

    # MULTI-TF 權重
    default_tf_weight = {"D": 1.0, "W": 1.5, "M": 2.0}

    # Broker 設定（沿用你的）
    SECURITIES_TRADER_IDS = [1440, 1470, 1480, 1650, 8440]
    broker_ratio_th = 0.05
    broker_zscore_th = 2.0
    broker_lookback = 20

    atr_multiplier = 3.0
    # -------------------------
    def __init__(self, *args, **kwargs):

        self.enable_pattern_model = kwargs.pop("enable_pattern_model", False)
        enable_pattern_factor = self.enable_pattern_model and kwargs.pop("enable_pattern", True)

        self.engine = FactorEngine(
            enable_pattern=enable_pattern_factor,
            enable_broker=kwargs.pop("enable_broker", True),
            enable_volume=kwargs.pop("enable_volume", True),
            enable_volatility=kwargs.pop("enable_volatility", True),

            weight_pattern=kwargs.pop("weight_pattern", 1.0),
            weight_broker=kwargs.pop("weight_broker", 1.0),
            weight_volume=kwargs.pop("weight_volume", 1),
            weight_vol_break=kwargs.pop("weight_vol_break", 1.0),

            buy_threshold=kwargs.pop("buy_threshold", 0.5),
            max_add=kwargs.pop("max_add", 3),
        )

        # Multi-TF
        # self.use_tf = kwargs.pop("use_tf", ["D", "W", "M"])
        self.use_tf = kwargs.pop("use_tf", ["D"])

        self.tf_weight = kwargs.pop("tf_weight", self.default_tf_weight)
        self.freq_modes = self.use_tf

        # Path
        self.stock_id = kwargs.get("stock_id", None)
        self.start_date = kwargs.get("start_date", None)
        self.end_date = kwargs.get("end_date", None)

        # Plot / predict workers
        self.workers_plot = kwargs.get("workers_plot", 4)
        self.workers_pred = kwargs.get("workers_pred", 4)

        base_dir = kwargs.get("base_dir", "/Users/meng-jutsai/Stock/FiveB/results/backtest/Alpha_v6")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join(base_dir, self.timestamp)

        self.plot_output_dir = os.path.join(self.base_dir, "plots")
        self.seg_output_dir = os.path.join(self.base_dir, "seg")
        os.makedirs(self.plot_output_dir, exist_ok=True)
        os.makedirs(self.seg_output_dir, exist_ok=True)

        super().__init__(*args, **kwargs)

        print(f"[Alpha_v6 initialized] stock={self.stock_id}")





    ############################################################
    #        Multi-process plot / predict
    ############################################################
    def _run_plot_parallel(self, trade_dates):

        tasks = []
        for d in trade_dates:
            d_str = d.strftime("%Y-%m-%d")
            start_360 = "1990-01-01"

            for tf in self.freq_modes:
                cmd = [
                    "python", self.plot_script,
                    "--stock_id", self.stock_id,
                    "--start_date", start_360,
                    "--end_date", d_str,
                    "--freq", tf,
                    "--output_dir", self.plot_output_dir,
                    "--date_folder", f"plots_{tf}"
                ]
                tasks.append({"cmd": cmd})

        print(f"[Alpha_v6] Start plotting {len(tasks)} images ...")

        with ProcessPoolExecutor(max_workers=self.workers_plot) as ex:
            for _ in as_completed([ex.submit(_plot_worker, t) for t in tasks]):
                pass

        print("[Alpha_v6] Plot done.")

    def _run_predict_parallel(self):

        for tf in self.freq_modes:
            image_dir = os.path.join(self.plot_output_dir, f"plots_{tf}")
            seg_csv = os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv")
            save_dir = os.path.join(self.seg_output_dir, f"seg_{tf}")
            os.makedirs(save_dir, exist_ok=True)

            cmd = [
                "python", self.predict_script,
                "--model_path", self.model_path,
                "--source_dir", image_dir,
                "--save_dir", save_dir,
                "--csv_file", seg_csv,
                "--parallel",
                "--workers", str(self.workers_pred)
            ]

            print(f"[Alpha_v6] Predict {tf} ...")
            subprocess.run(cmd, check=True)

        print("[Alpha_v6] Predict done.")


    ############################################################
    #           BROKER LOADER 
    ############################################################

    def _calc_tf_trend_factor(self, seg, date):
        """
        只在 freq_modes 有啟用對應 TF 時，才使用 W/M pattern
        W → +0.2
        M → +0.3
        """

        boost_W = 0.0
        boost_M = 0.0

        # seg 必須不是空
        if seg is None or seg.empty:
            return boost_W, boost_M

        # 只有在 freq_modes 有 W 才檢查 W 模式
        if "W" in self.freq_modes:
            df_W = seg[(seg["TF"] == "W")]
            df_W_today = df_W[df_W["Breakout_Date"].dt.date == date.date()]
            if not df_W_today.empty:
                boost_W = 0.2

        # 只有在 freq_modes 有 M 才檢查 M 模式
        if "M" in self.freq_modes:
            df_M = seg[(seg["TF"] == "M")]
            df_M_today = df_M[df_M["Breakout_Date"].dt.date == date.date()]
            if not df_M_today.empty:
                boost_M = 0.3

        return boost_W, boost_M




    ############################################################
    #           BROKER LOADER 
    ############################################################
    def _load_broker_flow(self, stock_id):

        conn = sqlite3.connect("/Users/meng-jutsai/Stock/FiveB/stock.db")

        q = f"""
            SELECT date, stock_id, SUM(net) AS net
            FROM tw_trading_daily_report
            WHERE stock_id = '{stock_id}'
            AND securities_trader_id IN ({",".join(map(str, self.SECURITIES_TRADER_IDS))})
            AND date BETWEEN '{self.start_date}' AND '{self.end_date}'
            GROUP BY date, stock_id
            ORDER BY date
        """

        df = pd.read_sql_query(q, conn, parse_dates=["date"])
        conn.close()
        if df.empty:
            return None

        # 1. lot 數
        df["net_lots"] = df["net"] / 1000.0

        # 2. 拆成買超 / 賣超（負值不進買均值）
        df["buy_lots"]  = df["net_lots"].clip(lower=0)
        df["sell_lots"] = (-df["net_lots"]).clip(lower=0)

        # 3. rolling mean 只用 buy lots

        df["buy_ema_fast"] = df["buy_lots"].ewm(span=10, adjust=False).mean()
        df["buy_ema_slow"] = df["buy_lots"].ewm(span=30, adjust=False).mean()
        df["strength"] = df["buy_ema_fast"] / (df["buy_ema_slow"] + 1e-9)

        mean = df["buy_lots"].ewm(span=30, adjust=False).mean()
        std = df["buy_lots"].ewm(span=30, adjust=False).std()
        df["z_buy"] = (df["buy_lots"] - mean) / (std + 1e-9)


        return df[[
            "date", "net_lots", "buy_lots", "sell_lots",
            "buy_ema_fast", "buy_ema_slow",
            "strength", "z_buy"
        ]]



    ############################################################
    #           BROKER MOMENTUM & ACCELERATION
    ############################################################
    def _calc_broker_momentum(self, sp, idx, window=10):
        if idx < window:
            return 0.0

        y = sp["buy_lots"].iloc[idx-window:idx].values
        if np.all(y == 0):
            return 0.0

        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope


    def _calc_broker_accel(self, sp, idx, window=10):
        if idx < window + 1:
            return 0.0

        mom_today = self._calc_broker_momentum(sp, idx, window)
        mom_prev  = self._calc_broker_momentum(sp, idx-1, window)
        return mom_today - mom_prev


    ############################################################
    #       BROKER LOOKBACK FACTOR (UPGRADED v4)
    ############################################################
    def _calc_broker_lookback_factor(self, sp, idx, window=5):

        if idx < 20:
            return 0.0

        buy_lots  = sp.loc[idx, "buy_lots"]
        z_buy     = sp.loc[idx, "z_buy"]

        slope     = self._calc_broker_momentum(sp, idx, window=10)
        accel     = self._calc_broker_accel(sp, idx, window=10)

        return self.engine.broker_factor(
            sp.loc[idx, "strength"],
            sp.loc[idx, "z_buy"],
            slope,
            accel
        )


    # ------------------------------------------------------
    # 輔助函數：計算 ATR
    # ------------------------------------------------------
    def _calculate_atr(self, df, window=14):
        # Calculate True Range (TR)
        df['TR'] = np.maximum(
            df['max'] - df['min'],
            np.maximum(abs(df['max'] - df['close'].shift(1)), abs(df['min'] - df['close'].shift(1)))
        )
        # Calculate Average True Range (ATR)
        df['ATR'] = df['TR'].rolling(window=window).mean()
        return df

    ############################################################
    # Volume / Volatility Features
    ############################################################
    def _calc_volume_ema(self, sp, idx, fast=5, slow=20):
        if idx < slow:
            return 0, 0
        vol_fast = sp["Trading_Volume"].ewm(span=fast, adjust=False).mean().iloc[idx]
        vol_slow = sp["Trading_Volume"].ewm(span=slow, adjust=False).mean().iloc[idx]
        return vol_fast, vol_slow



    def _calc_volatility_ema(self, sp, idx, fast=5, slow=20):
        if idx < slow:
            return 0, 0
        vola_fast = sp["close"].pct_change().abs().ewm(span=fast, adjust=False).mean().iloc[idx]
        vola_slow = sp["close"].pct_change().abs().ewm(span=slow, adjust=False).mean().iloc[idx]
        return vola_fast, vola_slow


    ############################################################
    #      Multi-TF Segmentation Loading / Conditions
    ############################################################
    def _extract_file_date(self, filename):
        parts = os.path.basename(filename).split("_")
        if len(parts) < 4:
            return None
        try:
            return datetime.strptime(parts[-3], "%Y-%m-%d").date()
        except:
            return None

    def _load_seg_multi(self, stock_id):
        dfs = []
        for tf in self.freq_modes:
            p = os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv")

            if not os.path.exists(p):
                continue

            df = pd.read_csv(p)
            df["TF"] = tf
            df["Breakout_Date"] = pd.to_datetime(df["Breakout_Date"])
            df["file_date"] = df["File"].apply(self._extract_file_date)
            df["stock_id"] = df["File"].apply(lambda x: x.split("_")[0])
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        return df_all[df_all["stock_id"] == str(stock_id)]

    def _is_breakout_tf(self, breakout_date, file_date, tf):
        if breakout_date is None or file_date is None:
            return False
        if tf == "D":
            return breakout_date == file_date
        elif tf == "W":
            return (file_date - pd.Timedelta(days=6) <= breakout_date <= file_date)
        elif tf == "M":
            return (file_date.replace(day=1) <= breakout_date <= file_date)
        return False



    def _check_multi_factor_exit(self, trade: dict, current_factors: dict) -> tuple[Union[str, None], Union[float, None]]:
        
        """
        根據持倉資訊、當日因子，以及進場主要因子，決定出場理由和出場價格。
        新的邏輯：Pilot 倉位（QuietAcc 進場）對籌碼反轉更敏感。
        """
        
        # 取得當日價格/因子
        low = current_factors['low']
        high = current_factors['high']
        price = current_factors['price']
        net_lots = current_factors['net_lots']
        f_broker = current_factors['f_broker']
        entry_type = trade.get("entry_type", "FULL")
        main_entry_factor = trade.get("main_entry_factor", "UNKNOWN") # 取得主要進場因子
        
        exit_reason = None
        exit_price = None

        # 1. 🛑 硬性止損 (Stop Loss) - 所有倉位共用
        if low <= trade["stop_price"]:
            exit_reason = "STOP_LOSS" 
            exit_price = trade["stop_price"]
            return exit_reason, exit_price
        
        # 2. 🎯 獲利了結檢查 (Take Profit) - 僅適用於型態/突破帶來的目標價
        # 只有當進場因子是 Pattern 相關時，目標價才有效
        if main_entry_factor == 'pattern': 
            if trade.get("target2") is not None and high >= trade["target2"]:
                exit_reason = "TP2"
                exit_price = trade["target2"]
                return exit_reason, exit_price

            elif trade.get("target1") is not None and high >= trade["target1"]:
                # 這裡可以加入只平倉一半的邏輯，但為了簡化先全平
                exit_reason = "TP1"
                exit_price = trade["target1"]
                return exit_reason, exit_price
                
        # 3. 📉 籌碼反轉出場 (Broker Factor Exit) - 根據進場因子調整敏感度
        
        # 籌碼敏感度閾值：
        # 如果是 QuietAcc 相關因子進場 (主要因子是 quiet_acc 或 broker)，則更敏感
        if main_entry_factor == "broker":
            broker_exit_th = 0.3
        else:
            broker_exit_th = 0.1

        # 如果主因子是 Price trend，但 f_price_short > 0.6 → 反轉出場
        if main_entry_factor == "price_long" and current_factors["f_price_short"] > 0.6:
            exit_reason = "PRICE_REVERSAL"
            exit_price = price
            return exit_reason, exit_price



        # 出場條件：當日淨賣量 AND 籌碼因子低於閾值
        if net_lots < 0 and f_broker < broker_exit_th: 
            exit_reason = f"BROKER_REVERSE_{main_entry_factor.upper()}"
            exit_price = price
            return exit_reason, exit_price
            
        # 5. Factor weakening exit（與 dynamic threshold 對齊）
        if current_factors.get("factor_downtrend", 0) == 1:
            return "FACTOR_WEAKEN", price


        return None, None



    def _calc_dynamic_threshold(self, sp, i, lookback=60, q=0.8):
        """
        動態分位數門檻（Quantile Threshold）
        最近 N 天 total_factor 的第 q 分位數
        """
        if i < lookback:
            # 初期資料不足 → 進場更保守
            return 0.7

        hist = sp["total_factor_score"].iloc[i-lookback:i]
        th = hist.quantile(q)

        # 避免極端情況（全部都是 0 之類）
        th = float(np.clip(th, -0.5, 0.95))
        return th

    def _calc_zscore_threshold(self, sp, i, lookback=20, z_th=1.0):
        if i < lookback:
            return 0.7

        hist = sp["total_factor_score"].iloc[i-lookback:i]
        mean = hist.mean()
        std  = hist.std() + 1e-9

        z_factor = (sp["total_factor_score"].iloc[i] - mean) / std

        return z_factor, (z_factor > z_th)


    def _calc_mean_surge(self, sp, i, lookback=20, surge_ratio=0.2):
        if i < lookback:
            return False

        hist = sp["total_factor_score"].iloc[i-lookback:i]
        th = hist.mean() * (1 + surge_ratio)

        return sp["total_factor_score"].iloc[i] > th


    ############################################################
    #                     MAIN STRATEGY LOGIC
    ############################################################

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs):
        """
        實作統一的因子綜合評分進場，並根據進場時的主導因子執行多情境出場。
        """

        stock_id = self.stock_id
        
        # Pre-processing
        sp = stock_price.copy()
        sp["date"] = pd.to_datetime(sp["date"])
        sp = sp.sort_values("date").reset_index(drop=True)

        seg = pd.DataFrame()
        broker_df = None

        if self.enable_pattern_model:
            # 1. Segmentation 運行
            need_predict = any(
                not os.path.exists(os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv"))
                for tf in self.freq_modes
            )
            if need_predict:
                print("[Alpha_v6] Running plot + segmentation ...")
                sp_tmp = stock_price.copy()
                sp_tmp["date"] = pd.to_datetime(sp_tmp["date"])
                self._run_plot_parallel(sp_tmp["date"].tolist())
                self._run_predict_parallel()

            seg = self._load_seg_multi(stock_id)


        broker_df = self._load_broker_flow(stock_id)
        sp = sp.merge(broker_df, on="date", how="left")

        # 連買天數
        sp["is_net_buy"] = sp["net_lots"] > 0
        sp["consecutive_net_buy_days"] = (
            sp.groupby((sp["is_net_buy"] == False).cumsum())["is_net_buy"].cumsum()
        )
        sp.loc[~sp["is_net_buy"], "consecutive_net_buy_days"] = 0

        # ATR
        sp = self._calculate_atr(sp, window=14)

        # 回測欄位
        sp["signal"] = 0
        sp["position"] = 0

        active = []    # 持倉池
        records = []
        conn = sqlite3.connect("/Users/meng-jutsai/Stock/FiveB/stock.db")
        
        # 初始化 Factor Weights (確保進場和歸因使用一致權重)
        self.factor_weights = {
            'pattern': self.engine.w_pattern,
            'broker': self.engine.w_broker,
            'vol_long': self.engine.w_volume,
            'vol_short': self.engine.w_volume_short,
            'price': 1,
            'volume': self.engine.w_volume
        }


        # =========================================================
        # 主要迴圈
        # =========================================================
        for i, row in sp.iterrows():

            date = row["date"]
            px = row["close"]
            high = row["max"]
            low = row["min"]
            current_pos = sum(t["position"] for t in active)

            # 非交易日/數據不足跳過
            q = pd.read_sql_query(
                "SELECT is_trading FROM tw_trading_calendar WHERE date=?",
                conn, params=(date.strftime("%Y-%m-%d"),)
            )
            if q.empty or q.iloc[0, 0] == 0 or i < 20:
                sp.loc[i, "position"] = current_pos
                continue

            # =====================================================
            # 每日因子計算 & Breakout 偵測
            # =====================================================


            # ------------------------
            # Price Factors (5 modules)
            # ------------------------
            trend_long, trend_short = self.engine.price_trend_factors(sp, i)
            pull_long, pull_short = self.engine.pullback_factors(sp, i)
            brk_long, brk_short = self.engine.breakout_breakdown_factors(sp, i)
            box_long, box_short = self.engine.price_box_factors(sp, i)
            rs_long, rs_short = self.engine.relative_strength_factors(
                sp, i, index_close=sp["close"]
            )

            f_price_long, f_price_short = self.engine.combined_price_factors(
                trend_long, trend_short,
                pull_long, pull_short,
                brk_long, brk_short,
                box_long, box_short,
                rs_long, rs_short
            )

            f_price_long, f_price_short, f_price_net = self.engine.normalize_price_factors(f_price_long, f_price_short)
            # 量
            vol_fast, vol_slow = self._calc_volume_ema(sp, i)
            vol_today, slow_ema = self.engine._calc_vol_breakout(sp, i)

            f_vol_long = self.engine.volume_long_factor(
                sp, i, vol_fast, vol_slow, vol_today, slow_ema
            )
            f_vol_short = self.engine.volume_short_factor(sp, i)

            f_vol_long_final, f_vol_short_final, f_vol_net = \
                self.engine.normalize_volume_factors(f_vol_long, f_vol_short)



            vola_fast, vola_slow = self._calc_volatility_ema(sp, i)



            # 分點
            f_broker = self._calc_broker_lookback_factor(sp, i, window=5)



            # 偵測 Pattern Breakout
            best_brk = None
            f_pattern = 0.0

            if self.enable_pattern_model and not seg.empty: # 檢查是否啟用
            

                # =============================================================
                # 只用 D 產生 breakout_signal
                # =============================================================
                df_D = seg[
                    (seg["TF"] == "D") &
                    (seg["Breakout_Date"].dt.date == date.date()) &
                    seg.apply(lambda r: self._is_breakout_tf(
                        r["Breakout_Date"].date(), r["file_date"], "D"), axis=1)
                ]
                df_D = df_D[df_D["Label"].isin(self.long_labels)]

                if not df_D.empty:
                    best_brk = df_D.loc[df_D["Fulfill_1st_Price"].idxmax()]
                    
                    base_class_w = self.pattern_class_weight.get(best_brk["Label"], 0.0)
                    base_tf_w = self.tf_weight["D"]

                    # D 的基礎 pattern 分數
                    f_pattern_D = base_class_w * base_tf_w

                else:
                    f_pattern_D = 0.0


                # =============================================================
                #  W / M 目標價（不觸發 BUY，只作為大級別 TP 參考）
                # =============================================================
                # 預設為 NaN
                sp.loc[i, "Weekly_target1"] = np.nan
                sp.loc[i, "Weekly_target2"] = np.nan
                sp.loc[i, "Monthly_target1"] = np.nan
                sp.loc[i, "Monthly_target2"] = np.nan

                # ------------------ Weekly (W) ------------------
                if "W" in self.freq_modes:
                    df_W = seg[
                        (seg["TF"] == "W") &
                        (seg["Breakout_Date"].dt.date == date.date()) &
                        seg.apply(lambda r: self._is_breakout_tf(
                            r["Breakout_Date"].date(), r["file_date"], "W"), axis=1)
                    ]
                    df_W = df_W[df_W["Label"].isin(self.long_labels)]

                    if not df_W.empty:
                        w_brk = df_W.loc[df_W["Fulfill_1st_Price"].idxmax()]
                        sp.loc[i, "Weekly_target1"] = w_brk["Fulfill_1st_Price"]
                        sp.loc[i, "Weekly_target2"] = w_brk["Fulfill_2nd_Price"]

                # ------------------ Monthly (M) ------------------
                if "M" in self.freq_modes:
                    df_M = seg[
                        (seg["TF"] == "M") &
                        (seg["Breakout_Date"].dt.date == date.date()) &
                        seg.apply(lambda r: self._is_breakout_tf(
                            r["Breakout_Date"].date(), r["file_date"], "M"), axis=1)
                    ]
                    df_M = df_M[df_M["Label"].isin(self.long_labels)]

                    if not df_M.empty:
                        m_brk = df_M.loc[df_M["Fulfill_1st_Price"].idxmax()]
                        sp.loc[i, "Monthly_target1"] = m_brk["Fulfill_1st_Price"]
                        sp.loc[i, "Monthly_target2"] = m_brk["Fulfill_2nd_Price"]

                # =============================================================
                #  W/M trend boosting（只有啟用的 TF 才會參與）
                # =============================================================
                boost_W, boost_M = self._calc_tf_trend_factor(seg, date)
                trend_multiplier = 1 + boost_W + boost_M

                f_pattern = f_pattern_D * trend_multiplier

                # ---------------------------------------------------------
                # 記錄 pattern 分數與 boosting
                # ---------------------------------------------------------
                sp.loc[i, "trend_boost_W"] = boost_W
                sp.loc[i, "trend_boost_M"] = boost_M
                sp.loc[i, "f_pattern"] = f_pattern

                # ---------------------------------------------------------
                # 寫入 D 線 target（真正 entry 用）
                # ---------------------------------------------------------
                if best_brk is not None:
                    sp.loc[i, "daily_target1"] = best_brk["Fulfill_1st_Price"]
                    sp.loc[i, "daily_target2"] = best_brk["Fulfill_2nd_Price"]
                else:
                    sp.loc[i, "daily_target1"] = np.nan
                    sp.loc[i, "daily_target2"] = np.nan



            # =====================================================
            # Multi-Factor 合成分數 & 歸因
            # =====================================================
            factor_dict = {
                "pattern": f_pattern,
                "broker": f_broker,
                "price": f_price_net,
                "volume": f_vol_net
            }



            total_score = self.engine.total_factor(
                f_pattern,
                f_broker,
                f_vol_net,
                vola_fast, vola_slow,
                f_price_net
            )


            # 找到貢獻最大的因子
            main_factor = max(
                factor_dict,
                key=lambda x: factor_dict[x] * self.factor_weights.get(x, 1)
            )

            # =============== 原始因子（raw 多空） ===============
            sp.loc[i, "f_pattern"] = f_pattern
            sp.loc[i, "f_broker"]  = f_broker
            sp.loc[i, "f_vol_long_raw"]  = f_vol_long_raw if "f_vol_long_raw" in locals() else f_vol_long
            sp.loc[i, "f_vol_short_raw"] = f_vol_short_raw if "f_vol_short_raw" in locals() else f_vol_short
            sp.loc[i, "f_price_long_raw"]  = f_price_long_raw if "f_price_long_raw" in locals() else f_price_long
            sp.loc[i, "f_price_short_raw"] = f_price_short_raw if "f_price_short_raw" in locals() else f_price_short

            # =============== Normalize 後（真正作用） ===============
            sp.loc[i, "f_price_net"] = f_price_net
            sp.loc[i, "f_vol_net"]   = f_vol_net

            # =============== 整體評分 & 主因子 ===============
            sp.loc[i, "total_factor_score"] = total_score
            sp.loc[i, "main_factor"]        = main_factor


            # 動態 threshold（Quantile Adaptive Threshold）
            dynamic_th = self._calc_dynamic_threshold(sp, i, lookback=20, q=0.8)
            sp.loc[i, "dynamic_th"] = dynamic_th

            # =====================================================
            # 執行出場邏輯 (多因子 + 主因子敏感度)
            # =====================================================
            current_factors = {
                'date': date,
                'price': px,
                'high': high,
                'low': low,
                'f_broker': f_broker,
                'f_vol_long': f_vol_long,
                'f_vol_short': f_vol_short,
                'net_lots': row["net_lots"], 
                'net_buy_days': row["consecutive_net_buy_days"],
                'ATR': row["ATR"],
                'f_price_long': f_price_long,
                'f_price_short': f_price_short
            }

            
            remove = []
            for k, t in enumerate(active):
                


                # 呼叫基於進場歸因的出場函數
                exit_reason, exit_price = self._check_multi_factor_exit(t, current_factors)
                
                if exit_reason:
                    # 記錄出場
                    records.append({
                        "date": date, "action": exit_reason, "price": exit_price,
                        "qty": -t["position"],
                        "entry_date": t["entry_date"], "entry_price": t["entry_price"],
                        "entry_type": t.get("entry_type", "FULL"),
                        "main_entry_factor": t["main_entry_factor"],
                        "main_exit_factor": exit_reason,
                    })

                    sp.loc[i, "signal"] -= t["position"]
                    remove.append(k)

            for k in sorted(remove, reverse=True):
                active.pop(k)


            # =====================================================
            # 接著看是否進場 (統一規則：總分達標)
            # =====================================================
            current_pos = sum(t["position"] for t in active)
            max_units = self.engine.add_units(total_score) # 決定總共能買多少單位
            qty_to_trade = max_units - current_pos # 還有多少單位可以加碼

            # 進場規則：總分達標 AND 還有可交易單位
            # if self.engine.should_buy(total_score) and qty_to_trade > 0:
            # if (total_score > dynamic_th) and (total_score > 0) and qty_to_trade > 0:

            # if self._calc_mean_surge(sp, i) and qty_to_trade > 0:


            z_factor, z_signal = self._calc_zscore_threshold(sp, i)
            sp.loc[i, "z_factor"]        = z_factor
            sp.loc[i, "z_signal"]        = z_signal

            if z_signal and total_score > 0 and qty_to_trade > 0:
                # 進場


                qty = 1 # 每次加碼 1 單位 (可調整)
                qty = min(qty, qty_to_trade)

                # 根據主要因子，決定目標價和止損類型
                target1, target2, stop_p = None, None, 0.0
                
                if main_factor == 'pattern' and best_brk is not None:
                    # Pattern 因子貢獻大：使用型態目標價，ATR 緊湊止損
                    target1 = best_brk["Fulfill_1st_Price"]
                    target2 = best_brk["Fulfill_2nd_Price"]
                    stop_p = px - self.atr_multiplier * row["ATR"] 
                    entry_type = "PATTERN_FULL"
                else:
                    # 其他因子貢獻大：無目標價，使用較寬鬆止損 (1.5倍 ATR)
                    stop_p = px - 1.5 * self.atr_multiplier * row["ATR"] 
                    entry_type = "FACTOR_FULL"

                action_type = "ADD-ON" if current_pos > 0 else "BUY"
                
                # 新倉位加入 active
                active.append(dict(
                    entry_date=date, entry_price=px, position=qty, 
                    target1=target1, target2=target2, stop_price=stop_p, 
                    TF="MultiFactor", entry_type=entry_type, 
                    main_entry_factor=main_factor
                ))

                # 記錄進場
                records.append({
                    "date": date, "action": action_type, "price": px,
                    "qty": qty, "entry_date": date, "entry_price": px,
                    "entry_type": entry_type,
                    "main_entry_factor": main_factor,
                    "total_score": total_score
                })

                sp.loc[i, "signal"] += qty

            sp.loc[i, "position"] = sum(t["position"] for t in active)

        # =========================================================
        # 迴圈結束後處理
        # =========================================================
        conn.close()
        self._trade_detail = pd.DataFrame(records)
        out_path = os.path.join(self.base_dir, "trade_records.csv")
        self._trade_detail.to_csv(out_path, index=False, encoding="utf-8-sig")

        sp["date"] = sp["date"].dt.strftime("%Y-%m-%d")
        print(f"[Alpha_v6] Trade records saved → {out_path}")
        return sp
