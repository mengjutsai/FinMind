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
        weight_volume=0.6,
        weight_volatility=0.6,

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

        self.buy_threshold = buy_threshold
        self.max_add = max_add

    # ------------------------------------------------------
    # Pattern Factor
    # ------------------------------------------------------
    def pattern_factor(self, ps, tf_weight):
        if not self.enable_pattern:
            return 0
        if pd.isna(ps):
            return 0
        return ps * tf_weight

    # ------------------------------------------------------
    # Broker Factor (沿用你 V11 的 scoring)
    # ------------------------------------------------------
    def broker_factor(self, broker_ratio, zscore):
        if not self.enable_broker:
            return 0

        if pd.isna(broker_ratio) or pd.isna(zscore):
            return 0

        if broker_ratio < 0 or zscore < 0:
            return 0

        ratio_score = min(1.0, broker_ratio / 0.20)
        zscore_score = min(1.0, zscore / 5.0)
        return 0.6 * ratio_score + 0.4 * zscore_score

    # ------------------------------------------------------
    # Volume Compression Factor
    # vol_ratio = vol_5 / vol_20  (越小越好)
    # score:
    #   <0.40 → 1.0
    #   0.40–0.60 → 0.5
    #   >0.60 → 0
    # ------------------------------------------------------
    def volume_compression_factor(self, vol_5, vol_20):
        if not self.enable_volume:
            return 0

        if vol_20 <= 0:
            return 0

        ratio = vol_5 / vol_20

        if ratio < 0.40:
            return 1.0
        elif ratio < 0.60:
            return 0.5
        else:
            return 0.0

    # ------------------------------------------------------
    # Volatility Compression
    # vola_ratio = std_5 / std_20  (越小越好)
    # score:
    #   <0.30 → 1.0
    #   0.30–0.50 → 0.5
    #   >0.50 → 0
    # ------------------------------------------------------
    def volatility_factor(self, std_5, std_20):
        if not self.enable_volatility:
            return 0

        if std_20 <= 0:
            return 0

        ratio = std_5 / std_20

        if ratio < 0.30:
            return 1.0
        elif ratio < 0.50:
            return 0.5
        else:
            return 0.0

    # ------------------------------------------------------
    # Total Factor
    # ------------------------------------------------------
    def total_factor(self, f_pattern, f_broker, f_vol, f_volatility):
        return (f_pattern * self.w_pattern +
                f_broker * self.w_broker +
                f_vol * self.w_volume +
                f_volatility * self.w_vol)

    # ------------------------------------------------------
    # Entry decision
    # ------------------------------------------------------
    def should_buy(self, total_factor):
        return total_factor >= self.buy_threshold

    def add_units(self, total_factor):
        units = 1 + int(total_factor)
        return min(units, self.max_add)



############################################################
#                    Alpha_v5 Strategy
############################################################

def _plot_worker(task):
    try:
        subprocess.run(task["cmd"], check=True)
        return True
    except Exception:
        return False


class Alpha_v5d1(Strategy):

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
    broker_lookback = 60

    # -------------------------
    def __init__(self, *args, **kwargs):

        self.engine = FactorEngine(
            enable_pattern=kwargs.pop("enable_pattern", True),
            enable_broker=kwargs.pop("enable_broker", True),
            enable_volume=kwargs.pop("enable_volume", True),
            enable_volatility=kwargs.pop("enable_volatility", True),

            weight_pattern=kwargs.pop("weight_pattern", 1.0),
            weight_broker=kwargs.pop("weight_broker", 1.0),
            weight_volume=kwargs.pop("weight_volume", 0.6),
            weight_volatility=kwargs.pop("weight_volatility", 0.6),

            buy_threshold=kwargs.pop("buy_threshold", 0.5),
            max_add=kwargs.pop("max_add", 3),
        )

        # Multi-TF
        self.use_tf = kwargs.pop("use_tf", ["D", "W", "M"])
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
    #           BROKER LOADER （完全沿用你 V11 的版本）
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

        df["net_lots"] = df["net"] / 1000.0
        df["broker_ratio"] = (
            df["net_lots"] /
            (df["net_lots"].rolling(self.broker_lookback).mean() + 1e-9)
        )

        df["zscore"] = df["net_lots"].rolling(self.broker_lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False
        )

        return df[["date", "net_lots", "broker_ratio", "zscore"]]


    def _calc_broker_lookback_factor(self, sp, idx, window=5):
        if idx < 2:
            return 0.0

        start = max(0, idx - window)
        hist = sp.iloc[start:idx]

        if hist.empty:
            return 0.0

        def _one_day_score(r):
            return self.engine.broker_factor(
                r.get("broker_ratio", np.nan),
                r.get("zscore", np.nan)
            )

        scores = hist.apply(_one_day_score, axis=1)
        return scores.mean()



    ############################################################
    # Volume / Volatility Features
    ############################################################
    def _calc_volume_features(self, sp, idx):
        if idx < 20:
            return 0, 0

        vol_5 = sp["Trading_Volume"].iloc[idx-5:idx].mean()
        vol_20 = sp["Trading_Volume"].iloc[idx-20:idx].mean()
        return vol_5, vol_20

    def _calc_volatility_features(self, sp, idx):
        if idx < 20:
            return 0, 0

        std_5 = sp["close"].iloc[idx-5:idx].std()
        std_20 = sp["close"].iloc[idx-20:idx].std()
        return std_5, std_20


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

    ############################################################
    #                     MAIN STRATEGY LOGIC
    ############################################################
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs):

        stock_id = self.stock_id

        # 若 segmentation 尚未存在 → 先做繪圖 + segmentation
        need_predict = any(
            not os.path.exists(os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv"))
            for tf in self.freq_modes
        )

        if need_predict:
            print("[Alpha_v5] Running plot + segmentation ...")
            sp_tmp = stock_price.copy()
            sp_tmp["date"] = pd.to_datetime(sp_tmp["date"])
            trade_dates = sp_tmp["date"].tolist()
            self._run_plot_parallel(trade_dates)
            self._run_predict_parallel()

        # segmentation 合併
        seg = self._load_seg_multi(stock_id)

        # broker factor
        broker_df = self._load_broker_flow(stock_id)

        # 價格資料
        sp = stock_price.copy()
        sp["date"] = pd.to_datetime(sp["date"])
        sp = sp.sort_values("date").reset_index(drop=True)
        sp = sp.merge(broker_df, on="date", how="left")

        # 回測欄位
        sp["signal"] = 0
        sp["position"] = 0

        active_trades = []
        records = []

        conn = sqlite3.connect("/Users/meng-jutsai/Stock/FiveB/stock.db")

        # -----------------------------
        # 主迴圈
        # -----------------------------
        for i, row in sp.iterrows():

            date = row["date"]
            price = row["close"]
            high = row["max"]
            low = row["min"]

            # 非交易日跳過
            q = pd.read_sql_query(
                "SELECT is_trading FROM tw_trading_calendar WHERE date=?",
                conn, params=(date.strftime("%Y-%m-%d"),)
            )
            if q.empty or q.iloc[0, 0] == 0:
                continue

            ############################################################
            # EXIT LOGIC
            ############################################################
            remove = []
            for k, t in enumerate(active_trades):

                exit_reason = None
                exit_price = None

                if low <= t["stop_price"]:
                    exit_reason = "STOP"
                    exit_price = t["stop_price"]

                elif high >= t["target2"]:
                    exit_reason = "TP2"
                    exit_price = t["target2"]

                elif high >= t["target1"]:
                    exit_reason = "TP1"
                    exit_price = t["target1"]

                if exit_reason:
                    qty = t["position"]

                    records.append({
                        "date": date,
                        "action": exit_reason,
                        "price": exit_price,
                        "qty": -qty,
                        "entry_date": t["entry_date"],
                        "entry_price": t["entry_price"],
                        "target1": t["target1"],
                        "target2": t["target2"],
                        "stop_price": t["stop_price"],
                        "TF": t.get("TF"),
                        "Label": t.get("Label"),
                        "Pattern_Score": t.get("Pattern_Score"),
                        "File": t.get("File"),
                    })

                    sp.loc[i, "signal"] -= qty
                    sp.loc[i, "exit_price"] = exit_price
                    sp.loc[i, "exit_reason"] = exit_reason


                    remove.append(k)

            for k in sorted(remove, reverse=True):
                active_trades.pop(k)

            ############################################################
            # BUY LOGIC（型態突破）
            ############################################################
            df_today = seg[
                (seg["Breakout_Date"].dt.date == date.date()) &
                seg.apply(lambda r: self._is_breakout_tf(
                    r["Breakout_Date"].date(), r["file_date"], r["TF"]), axis=1)
            ]

            df_today = df_today[df_today["Label"].isin(self.long_labels)]
            print(df_today)
            if not df_today.empty:

                best = df_today.loc[df_today["Fulfill_1st_Price"].idxmax()]

                ########################################################
                # 計算因子
                ########################################################
                f_pattern = self.engine.pattern_factor(
                    best["Pattern_Score"],
                    self.tf_weight[best["TF"]]
                )


                f_broker = self._calc_broker_lookback_factor(sp, i, window=5)


                # Volume compression
                vol_5, vol_20 = self._calc_volume_features(sp, i)
                f_vol = self.engine.volume_compression_factor(vol_5, vol_20)

                # Volatility compression
                std_5, std_20 = self._calc_volatility_features(sp, i)
                f_vola = self.engine.volatility_factor(std_5, std_20)



                total_f = self.engine.total_factor(f_pattern, f_broker, f_vol, f_vola)

                print("Date: %s, alpha score = %.2f" % (date,total_f))
                ########################################################
                # 若未達 threshold → 不買
                ########################################################
                if not self.engine.should_buy(total_f):
                    print("don't buy")
                    continue

                # 買入
                add_units = self.engine.add_units(total_f)

                p1 = best["Fulfill_1st_Price"]
                p2 = best["Fulfill_2nd_Price"]
                stop_price = price * (1 - 0.10)  # 固定 10%

                active_trades.append(dict(
                    entry_date=date,
                    entry_price=price,
                    position=add_units,
                    target1=p1,
                    target2=p2,
                    stop_price=stop_price,
                    TF=best["TF"],
                    Label=best["Label"],
                    Pattern_Score=best["Pattern_Score"],
                    File=best["File"]
                ))


                sp.loc[i, ["signal", "target1", "target2", "stop_price"]] = [
                    add_units, p1, p2, stop_price
                ]

                records.append({
                    "date": date,
                    "action": "BUY",
                    "price": price,
                    "qty": add_units,
                    "entry_date": date,
                    "entry_price": price,
                    "target1": p1,
                    "target2": p2,
                    "stop_price": stop_price,
                    "TF": best["TF"],
                    "Label": best["Label"],
                    "Pattern_Score": best["Pattern_Score"],
                    "File": best["File"],
                    "total_factor": total_f,
                    "pattern_factor": f_pattern,
                    "broker_factor": f_broker
                })

            sp.loc[i, "position"] = sum(t["position"] for t in active_trades)

        conn.close()

        self._trade_detail = pd.DataFrame(records)
        out = os.path.join(self.base_dir, "trade_records.csv")
        self._trade_detail.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[Alpha_v5] Trade records saved → {out}")

        sp["date"] = sp["date"].dt.strftime("%Y-%m-%d")
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
