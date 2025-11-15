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
    cmd = task["cmd"]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        return False


class Alpha_v3(Strategy):
    """
    改良版本：
    1. 先平行繪圖
    2. 再平行 YOLO segmentation 推論
    3. 產出 seg_results.csv
    4. 策略回測完全吃 seg_results.csv，不再每日跑 YOLO
    """

    plot_script = "/Users/meng-jutsai/Stock/FiveB/script/plot_from_sql.py"
    predict_script = "/Users/meng-jutsai/Stock/FiveB/script/predict_seg.py"

    # 你的 YOLO model
    model_path = "/Users/meng-jutsai/Stock/FiveB/runs/segment/yolov11m_seg_003/weights/best.pt"

    # 多/空 型態
    long_labels = {"Up-Triangle", "Up-W", "Up-Head-Shoulder-Bottom"}
    short_labels = {"Down-Triangle", "Down-M", "Down-Head-Shoulder-Top"}

    stop_loss_ratio = 0.10
    max_add = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FinMind 傳進來的基本參數
        self.stock_id = kwargs.get("stock_id", getattr(self, "stock_id", None))
        self.start_date = kwargs.get("start_date", getattr(self, "start_date", None))
        self.end_date = kwargs.get("end_date", getattr(self, "end_date", None))

        # 若 pattern 用的沒設定，使用 backtest 的 start/end
        self.pattern_start_date = kwargs.get("pattern_start_date", self.start_date)
        self.pattern_end_date = kwargs.get("pattern_end_date", self.end_date)

        if self.pattern_start_date is None or self.pattern_end_date is None:
            raise ValueError(f"[Alpha_v3] pattern_start_date or pattern_end_date is None. "
                            f"start={self.start_date}, end={self.end_date}")

        # workers
        self.workers_plot = kwargs.get("workers_plot", 4)
        self.workers_pred = kwargs.get("workers_pred", 4)

        base_dir = kwargs.get(
            "base_dir",
            "/Users/meng-jutsai/Stock/FiveB/results/backtest/alpha_v3"
        )

        # timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.base_dir = os.path.join(base_dir, self.timestamp)
        self.plot_output_dir = os.path.join(self.base_dir, "plots")
        self.seg_output_dir = os.path.join(self.base_dir, "image-seg")
        os.makedirs(self.plot_output_dir, exist_ok=True)
        os.makedirs(self.seg_output_dir, exist_ok=True)

        self.seg_csv_path = os.path.join(self.base_dir, "seg_results.csv")

        self.model_path = kwargs.get(
            "model_path",
            "/Users/meng-jutsai/Stock/FiveB/runs/segment/yolov11m_seg_003/weights/best.pt"
        )

        print(f"[Alpha_v3 initialized] stock={self.stock_id}, "
            f"start={self.pattern_start_date}, end={self.pattern_end_date}")


    # =============================================================
    # (A) 平行繪圖
    # =============================================================
    def run_plot_parallel(self, trade_dates):
        """
        平行畫圖：針對單一股票，對每一天 D 畫 (D-360 ~ D) 的 360-day 圖像。
        """

        tasks = []
        for d in trade_dates:
            d_str = d.strftime("%Y-%m-%d")
            start_360 = (d - pd.Timedelta(days=360)).strftime("%Y-%m-%d")

            out_dir = os.path.join(self.plot_output_dir, "image")
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                "python", self.plot_script,
                "--stock_id", self.stock_id,
                "--start_date", start_360,
                "--end_date", d_str,
                "--freq", "D",
                "--output_dir", self.plot_output_dir,
                "--date_folder", "skip"
            ]

            tasks.append({"cmd": cmd})

        print(f"[Alpha_v3] Start parallel plotting: {len(tasks)} images...")

        with ProcessPoolExecutor(max_workers=self.workers_plot) as ex:
            for _ in as_completed([ex.submit(_plot_worker, t) for t in tasks]):
                pass

        print("[Alpha_v3] Plot done.")


    # =============================================================
    # (B) 平行 YOLO segmentation predict
    # =============================================================
    def run_predict_parallel(self):
        source_dir = os.path.join(self.plot_output_dir, "image")

        cmd = [
            "python", self.predict_script,
            "--model_path", self.model_path,
            "--source_dir", source_dir,
            "--save_dir", self.seg_output_dir,
            "--csv_file", self.seg_csv_path,
            "--parallel",
            "--workers", str(self.workers_pred)
        ]

        print("[Alpha_v3] Start parallel YOLO segmentation...")
        subprocess.run(cmd, check=True)
        print("[Alpha_v3] Predict done.")



    # =============================================================
    # (C) 策略回測主程式 – 使用 seg_results.csv
    # =============================================================
    def create_trade_sign(self, stock_price: pd.DataFrame, **kwargs):

        stock_id = kwargs.get("stock_id", self.stock_id)

        # 若 seg_results.csv 尚未建立 → 先做產圖 + 推論

        if not os.path.exists(self.seg_csv_path):

            print(">>> seg_results.csv not found. Start batch plot + predict...")

            # 取全部日期
            sp = stock_price.copy()
            sp["date"] = pd.to_datetime(sp["date"])
            trade_dates = sp["date"].tolist()

            # 先畫所有圖
            self.run_plot_parallel(trade_dates)

            # 再跑 YOLO segmentation
            self.run_predict_parallel()


        # 讀取 seg_results.csv
        df_seg = pd.read_csv(self.seg_csv_path)

        print(self.seg_csv_path)
        print(df_seg)

        df_seg["Breakout_Date"] = pd.to_datetime(df_seg["Breakout_Date"], errors="coerce")

        if "stock_id" not in df_seg.columns:
            # raise ValueError("seg_results.csv 必須包含 stock_id 欄位")
            df_seg["stock_id"] = df_seg["File"].astype(str).apply(lambda x: x.split("_")[0])

        # 取該股票的全部訊號
        df_sig = df_seg[df_seg["stock_id"] == str(stock_id)]

        sp = stock_price.copy()
        sp["date"] = pd.to_datetime(sp["date"])
        sp = sp.sort_values("date").reset_index(drop=True)

        # 初始化欄位
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

            # 非交易日跳過
            q = pd.read_sql_query(
                "SELECT is_trading FROM tw_trading_calendar WHERE date=?",
                conn, params=(date.strftime("%Y-%m-%d"),)
            )
            if q.empty or q.iloc[0, 0] == 0:
                sp.loc[i, "position"] = sum(t["position"] for t in active_trades)
                continue

            # =========================================================
            # 1) 先處理出場邏輯
            # =========================================================
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
                    })

                    sp.loc[i, "signal"] -= qty
                    sp.loc[i, "exit_price"] = exit_price
                    sp.loc[i, "exit_reason"] = exit_reason

                    to_remove.append(idx2)

            for idx2 in sorted(to_remove, reverse=True):
                active_trades.pop(idx2)

            # =========================================================
            # 2) 當天是否出現突破訊號（來自 seg_results.csv）
            # =========================================================
            df_today = df_sig[df_sig["Breakout_Date"] == date]

            df_today_long = df_today[df_today["Label"].isin(self.long_labels)]

            if not df_today_long.empty:

                best = df_today_long.loc[df_today_long["Fulfill_1st_Price"].idxmax()]

                p1 = best["Fulfill_1st_Price"]
                p2 = best["Fulfill_2nd_Price"]
                stop_tmp = price * (1 - self.stop_loss_ratio)

                rr = (p1 - price) / (price - stop_tmp + 1e-9)
                add_units = min(max(int(rr), 1), self.max_add)

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
                    "target1": p1,
                    "target2": p2,
                    "stop_price": stop_tmp,
                    "rr": round(rr, 2)
                })

            sp.loc[i, "position"] = sum(t["position"] for t in active_trades)

        conn.close()
        sp["date"] = sp["date"].dt.strftime("%Y-%m-%d")

        self._trade_detail = pd.DataFrame(records)

        return sp
