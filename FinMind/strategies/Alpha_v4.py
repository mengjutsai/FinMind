import os
import subprocess
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from FinMind.strategies.base_sql import Strategy
from concurrent.futures import ProcessPoolExecutor, as_completed


def _plot_worker(task):
    import subprocess
    try:
        subprocess.run(task["cmd"], check=True)
        return True
    except Exception:
        return False


# ==========================================================
#                 Alpha_v4 (Multi-TF 共振)
# ==========================================================
class Alpha_v4(Strategy):

    plot_script = "/Users/meng-jutsai/Stock/FiveB/script/plot_from_sql.py"
    predict_script = "/Users/meng-jutsai/Stock/FiveB/script/predict_seg.py"
    model_path = "/Users/meng-jutsai/Stock/FiveB/runs/segment/yolov11m_seg_003/weights/best.pt"

    # 多/空 型態
    long_labels = {"Up-Triangle", "Up-W", "Up-Head-Shoulder-Bottom"}
    short_labels = {"Down-Triangle", "Down-M", "Down-Head-Shoulder-Top"}

    stop_loss_ratio = 0.10
    max_add = 3

    # 本次新增：多週期支援

    # freq_modes = ["D"]
    # freq_modes = ["W"]
    # freq_modes = ["D", "W", "M"]
    # tf_weight = {"D": 1.0, "W": 1.5, "M": 2.0}
    default_tf_weight = {"D": 1.0, "W": 1.5, "M": 2.0}

    # ----------------------------------------------------------
    def __init__(self, *args, **kwargs):

        self.use_tf = kwargs.pop("use_tf", ["D", "W", "M"])
        self.pattern_weight = kwargs.pop("pattern_weight", 1.0)
        self.volume_weight  = kwargs.pop("volume_weight", 1.0)
        self.tf_weight      = kwargs.pop("tf_weight", self.default_tf_weight)
        # self.shared_seg_dir = kwargs.pop("shared_seg_dir", None)
        # self.custom_base_dir = kwargs.pop("base_dir", None)

        # 僅啟用使用者指定的 TF
        self.freq_modes = self.use_tf

        super().__init__(*args, **kwargs)


        self.stock_id = kwargs.get("stock_id", getattr(self, "stock_id", None))
        self.start_date = kwargs.get("start_date", getattr(self, "start_date", None))
        self.end_date = kwargs.get("end_date", getattr(self, "end_date", None))

        self.pattern_start_date = kwargs.get("pattern_start_date", self.start_date)
        self.pattern_end_date = kwargs.get("pattern_end_date", self.end_date)

        if self.pattern_start_date is None or self.pattern_end_date is None:
            raise ValueError("pattern_start_date/pattern_end_date is None.")

        self.workers_plot = kwargs.get("workers_plot", 4)
        self.workers_pred = kwargs.get("workers_pred", 4)

        base_dir = kwargs.get(
            "base_dir",
            "/Users/meng-jutsai/Stock/FiveB/results/backtest/Alpha_v4"
        )

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join(base_dir, self.timestamp)

        self.plot_output_dir = os.path.join(self.base_dir, "plots")
        self.seg_output_dir = os.path.join(self.base_dir, "seg")

        os.makedirs(self.plot_output_dir, exist_ok=True)
        os.makedirs(self.seg_output_dir, exist_ok=True)

        print(f"[Alpha_v4 initialized] stock={self.stock_id}, "
              f"start={self.pattern_start_date}, end={self.pattern_end_date}")


    # ----------------------------------------------------------
    # 存交易紀錄
    # ----------------------------------------------------------

    def _save_trade_records(self, records):
        """將買賣紀錄輸出成 CSV"""
        if len(records) == 0:
            print("[Alpha_v4] No trade records to save.")
            return

        df = pd.DataFrame(records)

        out_path = os.path.join(self.base_dir, "trade_records.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[Alpha_v4] Trade records saved → {out_path}")




    # ----------------------------------------------------------
    # 取得 filename 的倒數第三段日期
    # ----------------------------------------------------------
    def _extract_filename_last_date(self, filename: str):
        base = os.path.basename(filename)
        parts = base.split("_")
        if len(parts) < 4:
            return None
        date_str = parts[-3]
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            return None



    def _is_breakout_in_tf_range(self, breakout_date, file_date, tf):
        """
        D: file_date == breakout_date
        W: breakout_date in [file_date - 6d  ~ file_date]
        M: breakout_date in [month_start ~ file_date]
        """

        if breakout_date is None or file_date is None:
            return False

        # --- Daily ---
        if tf == "D":
            return breakout_date == file_date

        # --- Weekly: 本週區間 ---
        elif tf == "W":
            week_end = file_date
            week_start = file_date - pd.Timedelta(days=6)
            return (week_start <= breakout_date <= week_end)

        # --- Monthly: 本月區間 ---
        elif tf == "M":
            month_end = file_date
            month_start = file_date.replace(day=1)
            return (month_start <= breakout_date <= month_end)

        return False


    # ----------------------------------------------------------
    # (A) Multi-TF 平行繪圖
    # ----------------------------------------------------------
    def run_plot_parallel(self, trade_dates):

        tasks = []
        for d in trade_dates:
            d_str = d.strftime("%Y-%m-%d")
            # start_360 = (d - pd.Timedelta(days=360)).strftime("%Y-%m-%d")
            start_360 = "1990-01-01"

            for tf in self.freq_modes:
                # out_dir = os.path.join(self.plot_output_dir, f"image")
                # os.makedirs(out_dir, exist_ok=True)

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

        print(f"[Alpha_v4] Start parallel plotting: {len(tasks)} images...")

        with ProcessPoolExecutor(max_workers=self.workers_plot) as ex:
            for _ in as_completed([ex.submit(_plot_worker, t) for t in tasks]):
                pass

        print("[Alpha_v4] Plot done.")

    # ----------------------------------------------------------
    # (B) Multi-TF segmentation
    # ----------------------------------------------------------
    def run_predict_parallel(self):

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

            print(f"[Alpha_v4] Predict {tf} ...")
            subprocess.run(cmd, check=True)

        print("[Alpha_v4] Predict done for D/W/M")

    # ----------------------------------------------------------
    # (C) 合併 Multi-TF seg results
    # ----------------------------------------------------------
    def _load_merge_seg_results(self, stock_id):

        dfs = []
        for tf in self.freq_modes:
            # path = os.path.join(
            #     self.shared_seg_dir if self.shared_seg_dir else self.seg_output_dir,
            #     f"seg_results_{tf}.csv"
            # )

            path = os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv")
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df["TF"] = tf
            df["stock_id"] = df["File"].astype(str).apply(lambda x: x.split("_")[0])
            df["Breakout_Date"] = pd.to_datetime(df["Breakout_Date"], errors="coerce")
            dfs.append(df)

        if len(dfs) == 0:
            raise ValueError("No segmentation result found.")

        df_all = pd.concat(dfs, ignore_index=True)

        return df_all[df_all["stock_id"] == str(stock_id)]


    # ----------------------------------------------------------
    # 成交量動能（Volume Momentum）
    # ----------------------------------------------------------
    def _check_volume_momentum(self, sp: pd.DataFrame, idx: int) -> bool:
        """
        成交量動能條件：
        Volume_today > MA20_volume * 1.3
        """

        if idx < 20:
            return False

        window = sp.iloc[idx-20:idx]
        vol_ma20 = window["Trading_Volume"].mean()
        today_vol = sp.iloc[idx]["Trading_Volume"]

        # 放量因子
        multiple = 1.3

        return today_vol > vol_ma20 * multiple


    # ----------------------------------------------------------
    # (D) 回測主程式：吃 Multi-TF 全部突破訊號
    # ----------------------------------------------------------
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs):

        stock_id = kwargs.get("stock_id", self.stock_id)

        # 檢查啟用的 timeframe 有沒有對應的 seg_results_{TF}.csv
        need_predict = False

        for tf in self.freq_modes:
            tf_csv = os.path.join(self.seg_output_dir, f"seg_results_{tf}.csv")

            if not os.path.exists(tf_csv):
                    need_predict = True

        if need_predict:
            print(">>> Multi-TF CSVs not found. Start batch plot + predict...")

            sp = stock_price.copy()
            sp["date"] = pd.to_datetime(sp["date"])
            trade_dates = sp["date"].tolist()

            self.run_plot_parallel(trade_dates)
            self.run_predict_parallel()


        # ------------------------------------------------------
        # Multi-TF 合併
        # ------------------------------------------------------
        df_sig = self._load_merge_seg_results(stock_id)

        # file_date = from filename
        df_sig["file_date"] = df_sig["File"].apply(self._extract_filename_last_date)

        # ------------------------------------------------------
        # 開始回測
        # ------------------------------------------------------
        sp = stock_price.copy()
        sp["date"] = pd.to_datetime(sp["date"])
        sp = sp.sort_values("date").reset_index(drop=True)

        sp["signal"] = 0
        sp["position"] = 0
        sp["target1"] = np.nan
        sp["target2"] = np.nan
        sp["stop_price"] = np.nan
        sp["exit_reason"] = ""
        sp["exit_price"] = np.nan

        active_trades = []
        records = []

        conn = sqlite3.connect("/Users/meng-jutsai/Stock/FiveB/stock.db")

        for i, row in sp.iterrows():

            date = row["date"]
            price = row["close"]
            high = row["max"]
            low = row["min"]

            # 非交易日
            q = pd.read_sql_query(
                "SELECT is_trading FROM tw_trading_calendar WHERE date=?",
                conn, params=(date.strftime("%Y-%m-%d"),)
            )
            if q.empty or q.iloc[0, 0] == 0:
                sp.loc[i, "position"] = sum(t["position"] for t in active_trades)
                continue

            # ------------------------------------------------------
            # 出場邏輯
            # ------------------------------------------------------
            to_remove = []
            for idx2, t in enumerate(active_trades):

                exit_reason = None
                exit_price = None

                if low <= t["stop_price"]:
                    exit_reason = "STOP"
                    exit_price = t["stop_price"]

                elif high >= t["target1"]:
                    exit_reason = "TP1"
                    exit_price = t["target1"]

                elif high >= t["target2"]:
                    exit_reason = "TP2"
                    exit_price = t["target2"]

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
                        "TF": t.get("TF", None),
                        "Label": t.get("Label", None),
                        "Pattern_Score": t.get("Pattern_Score", np.nan),
                        "File": t.get("File", None)
                    })




                    sp.loc[i, "signal"] -= qty
                    sp.loc[i, "exit_price"] = exit_price
                    sp.loc[i, "exit_reason"] = exit_reason
                    to_remove.append(idx2)

            for idx2 in sorted(to_remove, reverse=True):
                active_trades.pop(idx2)

            # ------------------------------------------------------
            # 當天 Multi-TF 突破訊號
            # (Breakout_Date == date) AND (file_date == date)
            # ------------------------------------------------------
            # df_today = df_sig[
            #     (df_sig["Breakout_Date"].dt.date == date.date()) &
            #     (df_sig["Breakout_Date"].dt.date == df_sig["file_date"])
            # ]


            df_today = df_sig[
                df_sig.apply(
                    lambda r: self._is_breakout_in_tf_range(
                        breakout_date = r["Breakout_Date"].date(),
                        file_date    = r["file_date"],
                        tf           = r["TF"]
                    ),
                    axis=1
                ) &
                (df_sig["Breakout_Date"].dt.date == date.date())
            ]
            
            df_today_long = df_today[df_today["Label"].isin(self.long_labels)]

            if not df_today_long.empty:

                # --------------------------------------------------
                # Multi-TF 加權強度（共振）
                # Σ (TF weight × Pattern Score)
                # --------------------------------------------------
                # (1) TF 共振加權

                df_today_long["TF_W"] = df_today_long["TF"].map(self.tf_weight)

                # 單一 Pattern Score（只乘一次）
                df_today_long["ps"] = df_today_long["Pattern_Score"]

                # Final pattern strength = PatternScore × TF Weight × Human weight
                df_today_long["wt_pattern"] = df_today_long["ps"] * df_today_long["TF_W"] * self.pattern_weight

                # Volume factor
                vol_factor = 1.0
                if self._check_volume_momentum(sp, i):
                    vol_factor = 1.0 + self.volume_weight

                df_today_long["wt_volume"] = vol_factor

                # Final Weight（完全不平方）
                df_today_long["WT"] = df_today_long["wt_pattern"] * df_today_long["wt_volume"]

                mtf_strength = df_today_long["WT"].sum()

                
                # 基本 1 單位 + 強度額外加碼
                add_units = 1 + int(mtf_strength)
                add_units = min(add_units, self.max_add)

                # 選最有利的突破（TP1最大）
                best = df_today_long.loc[df_today_long["Fulfill_1st_Price"].idxmax()]
                p1 = best["Fulfill_1st_Price"]
                p2 = best["Fulfill_2nd_Price"]

                stop_tmp = price * (1 - self.stop_loss_ratio)

                trade = dict(
                    entry_date=date,
                    entry_price=price,
                    target1=p1,
                    target2=p2,
                    stop_price=stop_tmp,
                    position=add_units
                )
                active_trades.append(trade)

                sp.loc[i, ["signal", "target1", "target2", "stop_price"]] = [
                    add_units, p1, p2, stop_tmp
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
                    "stop_price": stop_tmp,
                    "tf_strength": round(mtf_strength, 2),
                    "TF": best["TF"],
                    "Label": best["Label"],
                    "Pattern_Score": best.get("Pattern_Score", np.nan),
                    "File": best["File"]
                })


            sp.loc[i, "position"] = sum(t["position"] for t in active_trades)

        conn.close()

        sp["date"] = sp["date"].dt.strftime("%Y-%m-%d")
        self._trade_detail = pd.DataFrame(records)

        self._save_trade_records(records)

        return sp
