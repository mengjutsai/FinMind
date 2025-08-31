import os
import cv2
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.font_manager as fm
from datetime import datetime
from ultralytics import YOLO
from ta.momentum import StochasticOscillator
from FinMind.strategies.base_sql import Strategy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------
# 全域繪圖風格 (紅漲綠跌)
# -------------------
font_path = '/System/Library/Fonts/PingFang.ttc'
font_prop = fm.FontProperties(fname=font_path)

custom_style = mpf.make_mpf_style(
    base_mpf_style='charles',
    marketcolors=mpf.make_marketcolors(
        up='red', down='green',
        edge={'up': 'red', 'down': 'green'},
        wick={'up': 'red', 'down': 'green'},
        volume={'up': 'red', 'down': 'green'}
    ),
    rc={'font.sans-serif': font_prop.get_name()}
)


# -------------------
# KD 指標
# -------------------
def add_kd_indicators(df, n=9, d_n=9):
    kd = StochasticOscillator(
        high=df["max"], low=df["min"], close=df["close"],
        n=n, d_n=d_n, fillna=True
    )
    rsv = kd.stoch().fillna(50)
    k, d = np.zeros(len(df)), np.zeros(len(df))
    for i, r in enumerate(rsv):
        if i == 0:
            k[i], d[i] = 50, 50
        else:
            k[i] = k[i-1] * 2/3 + r / 3
            d[i] = d[i-1] * 2/3 + k[i] / 3
    df["K"], df["D"] = k, d
    return df


# -------------------
# MACD
# -------------------
def calculate_macd(df, short=12, long=26, signal_n=9):
    short_ema = df["close"].ewm(span=short, adjust=False).mean()
    long_ema = df["close"].ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_n, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


# -------------------
# Normalize (固定視窗大小)
# -------------------
def normalize_data(df, window=120):
    df = df.tail(window).copy()
    price_cols = ["open", "max", "min", "close"]

    # 找 normalization 的上下限
    min_p, max_p = df[price_cols].min().min(), df[price_cols].max().max()

    # 存原始價格，方便之後回算
    for c in price_cols:
        df[c + "_orig"] = df[c]

    # 做 0-1 normalization
    for c in price_cols:
        df[c] = (df[c] - min_p) / (max_p - min_p)

    # 成交量做 log normalize
    df["Trading_Volume_orig"] = df["Trading_Volume"]
    df["Trading_Volume"] = np.log1p(df["Trading_Volume"])

    # 回傳 norm 後的 df + scaling 參數
    return df, min_p, max_p

def denormalize(price_norm, min_p, max_p):
    return price_norm * (max_p - min_p) + min_p


def denormalize_data(df):
    """輸入 normalize_data 存下的 df，自動回復 open/high/low/close"""
    if "min_price" not in df.columns or "max_price" not in df.columns:
        raise ValueError("⚠️ 缺少 min_price / max_price 欄位，無法還原")

    min_p = df["min_price"].iloc[0]
    max_p = df["max_price"].iloc[0]

    for c in ["open", "max", "min", "close"]:
        df[c + "_recovered"] = df[c] * (max_p - min_p) + min_p

    return df
# -------------------
# 策略 (YOLO + K 線圖)
# -------------------
class TriangleBreakout_V3(Strategy):
    window: int = 360
    model_path: str = "/Users/meng-jutsai/Stock/pattern_detection/model/seg/v0.0.1.pt"
    label_names = [
        "Down-False-Breakout", "Down-Head-Shoulders-Top", "Down-M",
        "Down-Triangle", "Up-Head-Shoulders-Bottom", "Up-Triangle", "Up-W"
    ]

    long_patterns = {"Up-Triangle", "Up-W", "Up-Head-Shoulders-Bottom"}

    # 畫單一視窗的 K 線圖
    # 畫單一視窗的 K 線圖 (存在 raw/)
    def plot_stock_sliding(self, sub_df, stock_id, cur_date, output_dir):
        raw_img_dir = os.path.join(output_dir, "raw", "images")
        raw_csv_dir = os.path.join(output_dir, "raw", "csv")
        os.makedirs(raw_img_dir, exist_ok=True)
        os.makedirs(raw_csv_dir, exist_ok=True)

        sub_df, min_p, max_p = normalize_data(sub_df, window=len(sub_df))
        sub_df = add_kd_indicators(sub_df)
        macd, signal, hist = calculate_macd(sub_df)

        # 存 normalized data CSV
        date_str = pd.to_datetime(cur_date).strftime("%Y-%m-%d")
        norm_csv = os.path.join(raw_csv_dir, f"{stock_id}_{date_str}_norm.csv")
        sub_df["min_price"] = min_p
        sub_df["max_price"] = max_p
        sub_df.to_csv(norm_csv, index=False)

        hist_colors = ['green' if v < 0 else 'red' for v in hist]
        add_plots = [
            mpf.make_addplot(sub_df["K"], panel=2, color="blue"),
            mpf.make_addplot(sub_df["D"], panel=2, color="orange"),
            mpf.make_addplot(macd, panel=3, color="green"),
            mpf.make_addplot(signal, panel=3, color="red"),
            mpf.make_addplot(hist, panel=3, type="bar", color=hist_colors, alpha=0.5)
        ]

        kline_data = sub_df.rename(columns={
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume"
        })[["date", "Open", "High", "Low", "Close", "Volume"]]
        kline_data["date"] = pd.to_datetime(kline_data["date"])
        kline_data = kline_data.set_index("date")

        fname = f"{stock_id}_{date_str}_norm.png"
        fpath = os.path.join(raw_img_dir, fname)

        mpf.plot(kline_data, type="candle", volume=True, style=custom_style,
                 addplot=add_plots, savefig=fpath,
                 title=f"{stock_id} (Norm 0-1)", volume_panel=1,
                 ylabel="Norm Price", ylabel_lower="Log Volume", figratio=(16, 9))
        return fpath, min_p, max_p

    # 單圖 inference，會產生「帶框」的圖 & 存 reco csv
    def run_inference(self, fpath, sub_df, min_p, max_p, debug=False):
        model = YOLO(self.model_path)
        results = model.predict(source=fpath, save=False, verbose=debug)

        records = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ⚡ 調整輸出路徑：det 跟 raw 平行
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(fpath)))  # 回到 output_dir
        det_img_dir = os.path.join(base_dir, "det", "images")
        det_csv_dir = os.path.join(base_dir, "det", "csv")
        os.makedirs(det_img_dir, exist_ok=True)
        os.makedirs(det_csv_dir, exist_ok=True)

        for r in results:
            plotted_img = r.plot()
            fname_img = os.path.basename(fpath).replace("_norm.png", "_det.png")
            out_img_path = os.path.join(det_img_dir, fname_img)
            cv2.imwrite(out_img_path, plotted_img)

            img_height, img_width = plotted_img.shape[:2]
            bar_width = img_width / len(sub_df)

            for box in r.boxes:
                label_index = int(box.cls.item())
                conf = float(box.conf.item())
                label_name = model.names[label_index]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 四個角點
                corners = [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]
                corner_dates, corner_prices = [], []

                for (xx,yy) in corners:
                    # X→日期
                    idx = min(int(xx / bar_width), len(sub_df)-1)
                    date_at_x = sub_df.iloc[idx]["date"]

                    # Y→價格
                    norm_y = 1 - (yy / img_height)
                    price_at_y = norm_y * (max_p - min_p) + min_p

                    corner_dates.append(date_at_x)
                    corner_prices.append(price_at_y)

                records.append([
                    os.path.basename(r.path), label_name, conf,
                    x1,y1,x2,y2,
                    corner_dates[0], corner_prices[0],  # 左上
                    corner_dates[1], corner_prices[1],  # 右上
                    corner_dates[2], corner_prices[2],  # 左下
                    corner_dates[3], corner_prices[3],  # 右下
                    img_width, timestamp
                ])

        df = pd.DataFrame(records, columns=[
            'file','label','confidence',
            'x1','y1','x2','y2',
            'date_x1y1','price_x1y1',
            'date_x2y1','price_x2y1',
            'date_x1y2','price_x1y2',
            'date_x2y2','price_x2y2',
            'img_width','timestamp'
        ])

        # 存 CSV
        if not df.empty:
            fname_csv = os.path.basename(fpath).replace("_norm.png", "_det.csv")
            det_csv_path = os.path.join(det_csv_dir, fname_csv)
            df.to_csv(det_csv_path, index=False)

        return df



    # -------------------
    # 主要策略邏輯
    # -------------------
    def create_trade_sign(self, stock_price, **kwargs):
        stock_id = kwargs.get("stock_id", self.stock_id)
        img_dir  = kwargs.get("img_dir", f"./patterns/{stock_id}/images_normalized")

        # 🔑 這裡改：先用 kwargs，沒有的話 fallback 到 self.start_date / self.end_date
        start_date = kwargs.get("start_date", getattr(self, "start_date", None))
        end_date   = kwargs.get("end_date", getattr(self, "end_date", None))

        if start_date: start_date = pd.to_datetime(start_date)
        if end_date:   end_date   = pd.to_datetime(end_date)

        # 🚀 忽略 stock_price，直接重抓 extended 資料
        if start_date is not None and end_date is not None:
            min_date = start_date - pd.Timedelta(days=self.window)
            sp = self.data_loader.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=min_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            ).reset_index(drop=True)
        else:
            sp = stock_price.copy().reset_index(drop=True)

        sp["date"] = pd.to_datetime(sp["date"])
        sp["signal"] = 0

        # -----------------------------
        # 在 start_date~end_date 之間逐日做 [t-360, t]
        # -----------------------------
        for i, cur_date in enumerate(sp["date"]):
            # 如果有限制區間，跳過不在範圍內的日期
            if start_date and end_date:
                if cur_date < start_date or cur_date > end_date:
                    continue

            # window = [cur_date-360, cur_date]
            window_start = cur_date - pd.Timedelta(days=self.window)
            sub_df = sp[(sp["date"] > window_start) & (sp["date"] <= cur_date)].copy()

            # ⚠️ 新增：特別處理 start_date，確保跑 [start-360, start]
            if start_date and cur_date == start_date:
                window_start = start_date - pd.Timedelta(days=self.window)
                sub_df = sp[(sp["date"] > window_start) & (sp["date"] <= start_date)].copy()

            if len(sub_df) < self.window // 2:  # 避免資料不足
                continue

            # 繪圖 + YOLO 推論
            # fpath = self.plot_stock_sliding(sub_df, stock_id, cur_date, img_dir)
            # detections = self.run_inference(fpath, debug=False)
            
            fpath, min_p, max_p = self.plot_stock_sliding(sub_df, stock_id, cur_date, img_dir)
            detections = self.run_inference(fpath, sub_df, min_p, max_p, debug=False)

            if not detections.empty:
                # 找出多頭型態，且 x2 接近右邊界的框
                detections = detections[
                    (detections["label"].isin(self.long_patterns)) &
                    (detections["x2"] >= 0.85 * detections["img_width"])   # x2 > 90% 圖片寬度
                ]

                if not detections.empty:
                    sp.loc[sp["date"] == cur_date, "signal"] = 1

        print(sp)

        # ⚠️ 只保留 start_date ~ end_date 區間
        if start_date and end_date:
            sp = sp[(sp["date"] >= start_date) & (sp["date"] <= end_date)].reset_index(drop=True)

        # 🔑 強制轉回字串，避免 merge 出錯
        sp["date"] = sp["date"].dt.strftime("%Y-%m-%d")

        for col in ["CashEarningsDistribution", "StockEarningsDistribution"]:
            if col not in sp.columns:
                sp[col] = 0.0

        return sp
