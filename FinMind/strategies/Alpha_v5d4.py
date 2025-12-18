############################################################
#                    Alpha_v5                        #
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
        weight_volume=0.8,
        weight_volatility=0.6,
        weight_vol_break=1.0,

        # 全域判斷用
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
        self.w_vol = weight_volatility
        self.w_vol_break = weight_vol_break

        self.buy_threshold = buy_threshold
        self.max_add = max_add

    # ------------------------------------------------------
    # Pattern Factor
    # ------------------------------------------------------
    def pattern_factor(self, ps, tf_weight):
        if not self.enable_pattern or pd.isna(ps):
            return 0
        return ps * tf_weight

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

    ############################################################
    # Volume Compression（EMA Version）
    ############################################################
    def volume_compression_factor(self, vol_ema_fast, vol_ema_slow):
        if not self.enable_volume:
            return 0

        if vol_ema_slow <= 0:
            return 0

        r = vol_ema_fast / vol_ema_slow     # <1 越小越好

        # 連續化，用 1-r
        # r < 0.4 → score≈1
        # r=1 → score≈0
        score = np.tanh((1 - r) * 2)

        return max(0, score)

    ############################################################
    # Volatility Compression（EMA Version）
    ############################################################
    def volatility_factor(self, vola_ema_fast, vola_ema_slow):
        if not self.enable_volatility:
            return 0

        if vola_ema_slow <= 0:
            return 0

        r = vola_ema_fast / vola_ema_slow  # 越小越好

        score = np.tanh((1 - r) * 2)
        return max(0, score)

    ############################################################
    # Volume Breakout（EMA Version）
    ############################################################
    def volume_breakout_factor(self, vol_today, vol_slow_ema):
        if vol_slow_ema <= 0:
            return 0

        r = vol_today / vol_slow_ema  # 越大越好（突破）
        
        # r=1 → score≈0
        # r=2 → score≈1
        score = np.tanh((r - 1.0))

        return max(0, score)


    def _calc_vol_breakout(self, sp, idx, slow=20):
        if idx < slow:
            return 0, 0
        today = sp["Trading_Volume"].iloc[idx]
        slow_ema = sp["Trading_Volume"].ewm(span=slow, adjust=False).mean().iloc[idx]
        return today, slow_ema

    ############################################################
    # Total factor
    ############################################################
    def total_factor(self, f_pattern, f_broker, f_vol, f_vola, f_vol_break):
        return (
            f_pattern * self.w_pattern +
            f_broker * self.w_broker +
            f_vol * self.w_volume +
            f_vola * self.w_vol +
            f_vol_break * self.w_vol_break
        )

    ############################################################
    # Entry decision
    ############################################################
    def should_buy(self, total_factor):
        return total_factor >= self.buy_threshold

    def add_units(self, total_factor):
        return min(1 + int(total_factor), self.max_add)



############################################################
#                    Alpha_v5 Strategy
############################################################

def _plot_worker(task):
    try:
        subprocess.run(task["cmd"], check=True)
        return True
    except Exception:
        return False


class Alpha_v5d3(Strategy):

    # YOLO 模型與腳本
    plot_script = "/Users/meng-jutsai/Stock/FiveB/script/plot_from_sql.py"
    predict_script = "/Users/meng-jutsai/Stock/FiveB/script/predict_seg.py"
    model_path = "/Users/meng-jutsai/Stock/FiveB/runs/segment/yolov11m_seg_003/weights/best.pt"

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

        self.engine = FactorEngine(
            enable_pattern=kwargs.pop("enable_pattern", True),
            enable_broker=kwargs.pop("enable_broker", True),
            enable_volume=kwargs.pop("enable_volume", True),
            enable_volatility=kwargs.pop("enable_volatility", True),

            weight_pattern=kwargs.pop("weight_pattern", 1.0),
            weight_broker=kwargs.pop("weight_broker", 1.0),
            weight_volume=kwargs.pop("weight_volume", 1),
            weight_volatility=kwargs.pop("weight_volatility", 1),
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

        base_dir = kwargs.get("base_dir", "/Users/meng-jutsai/Stock/FiveB/results/backtest/Alpha_v5")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join(base_dir, self.timestamp)

        self.plot_output_dir = os.path.join(self.base_dir, "plots")
        self.seg_output_dir = os.path.join(self.base_dir, "seg")
        os.makedirs(self.plot_output_dir, exist_ok=True)
        os.makedirs(self.seg_output_dir, exist_ok=True)

        super().__init__(*args, **kwargs)

        print(f"[Alpha_v5 initialized] stock={self.stock_id}")

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
        f_vol_break = current_factors['f_vol_break']
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



        # 出場條件：當日淨賣量 AND 籌碼因子低於閾值
        if net_lots < 0 and f_broker < broker_exit_th: 
            exit_reason = f"BROKER_REVERSE_{main_entry_factor.upper()}"
            exit_price = price
            return exit_reason, exit_price
            
        # 4. 🚀 量能衰竭/反轉出場 (Volume Exhaustion)
        # 條件：爆量 (Vol_Break > 2.0) 且當日淨賣量 (net_lots < 0)
        if f_vol_break > 2.0 and net_lots < 0: 
            exit_reason = "VOL_EXHAUST"
            exit_price = price
            return exit_reason, exit_price

        return None, None


    ############################################################
    #                     MAIN STRATEGY LOGIC
    ############################################################

    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs):
        """
        實作統一的因子綜合評分進場，並根據進場時的主導因子執行多情境出場。
        """

        stock_id = self.stock_id
        
        # ---------------------------------------------------------
        # 數據準備 (保持不變)
        # ---------------------------------------------------------

        # 1. Segmentation 運行
        need_predict = any(
            not os.path.exists(os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv"))
            for tf in self.freq_modes
        )
        if need_predict:
            print("[Alpha_v5] Running plot + segmentation ...")
            sp_tmp = stock_price.copy()
            sp_tmp["date"] = pd.to_datetime(sp_tmp["date"])
            self._run_plot_parallel(sp_tmp["date"].tolist())
            self._run_predict_parallel()

        # 2. 載入 Seg/Broker
        seg = self._load_seg_multi(stock_id)
        broker_df = self._load_broker_flow(stock_id)

        # 3. 合併與預處理
        sp = stock_price.copy()
        sp["date"] = pd.to_datetime(sp["date"])
        sp = sp.sort_values("date").reset_index(drop=True)
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
            'vol': self.engine.w_volume,
            'vola': self.engine.w_volatility,
            'vol_break': self.engine.w_vol_break,
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
            f_broker = self._calc_broker_lookback_factor(sp, i, window=5)

            vol_fast, vol_slow = self._calc_volume_ema(sp, i)
            f_vol = self.engine.volume_compression_factor(vol_fast, vol_slow)


            vola_fast, vola_slow = self._calc_volatility_ema(sp, i)
            f_vola = self.engine.volatility_factor(vola_fast, vola_slow)


            vol_today, slow_ema = self.engine._calc_vol_breakout(sp, i)
            f_vol_break = self.engine.volume_breakout_factor(vol_today, slow_ema)


            
            # 偵測 Pattern Breakout
            best_brk = None
            f_pattern = 0.0
            df_today = seg[
                (seg["Breakout_Date"].dt.date == date.date()) &
                seg.apply(lambda r: self._is_breakout_tf(
                    r["Breakout_Date"].date(), r["file_date"], r["TF"]), axis=1)
            ]
            df_today = df_today[df_today["Label"].isin(self.long_labels)]

            if not df_today.empty:
                best_brk = df_today.loc[df_today["Fulfill_1st_Price"].idxmax()]
                f_pattern = self.engine.pattern_factor(
                    best_brk["Pattern_Score"], self.tf_weight[best_brk["TF"]]
                )

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
                'pattern': f_pattern, 'broker': f_broker, 'vol': f_vol,
                'vola': f_vola, 'vol_break': f_vol_break
            }

            # 計算總分數 (加權和)
            total_score = sum(
                factor_dict[k] * self.factor_weights.get(k, 0)
                for k in factor_dict
            )

            # 找到貢獻最大的因子
            main_factor = max(
                factor_dict, 
                key=lambda x: factor_dict[x] * self.factor_weights.get(x, 0)
            )
            

            sp.loc[i, "f_pattern"]     = f_pattern
            sp.loc[i, "f_broker"]      = f_broker
            sp.loc[i, "f_vol"]         = f_vol
            sp.loc[i, "f_vola"]        = f_vola
            sp.loc[i, "f_vol_break"]   = f_vol_break

            # 記錄當日因子分數
            sp.loc[i, "total_factor_score"] = total_score
            sp.loc[i, "main_factor"] = main_factor

            # =====================================================
            # 執行出場邏輯 (多因子 + 主因子敏感度)
            # =====================================================
            current_factors = {
                'date': date, 'price': px, 'high': high, 'low': low,
                'f_broker': f_broker, 'f_vol': f_vol, 'f_vola': f_vola,
                'f_vol_break': f_vol_break, 'net_lots': row["net_lots"], 
                'net_buy_days': row["consecutive_net_buy_days"], 'ATR': row["ATR"]
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
            if self.engine.should_buy(total_score) and qty_to_trade > 0:

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
        print(f"[Alpha_v5] Trade records saved → {out_path}")
        return sp



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

        print(f"[Alpha_v5] Start plotting {len(tasks)} images ...")

        with ProcessPoolExecutor(max_workers=self.workers_plot) as ex:
            for _ in as_completed([ex.submit(_plot_worker, t) for t in tasks]):
                pass

        print("[Alpha_v5] Plot done.")

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

            print(f"[Alpha_v5] Predict {tf} ...")
            subprocess.run(cmd, check=True)

        print("[Alpha_v5] Predict done.")
