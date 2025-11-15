import os, sys, sqlite3
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from FinMind.strategies.base_sql import Strategy

import argparse, csv, os
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from typing import Optional
import pandas as pd
import json
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from datetime import datetime
tmp_label = datetime.now().strftime("%Y%m%d_%H%M%S")


# === 動態載入你的 predict_seg ===
sys.path.append("/Users/meng-jutsai/Stock/FiveB/script")
from predict_seg import run_inference

global_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 載入中文字型（macOS）
font_path = "/System/Library/Fonts/PingFang.ttc"
font_pil = ImageFont.truetype(font_path, 12)  # 字體大小16可自行調整

def put_chinese_text(img_bgr, text, pos, color=(0,0,0), font=font_pil):
    """在 OpenCV 圖上畫出中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color[::-1])  # color 要轉成 RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === Global Debug Toggles ===
SHOW_BOX = False          # 是否顯示 YOLO bounding box
SHOW_SEGMENT = False      # 是否顯示 segmentation 多邊形與填色

# 單一/多類皆可（若模型內含 names 會覆蓋）
label_names = ["Up-Triangle", "Down-Triangle", "W", "M"]

## conversion

def load_date_list(meta_path: str) -> list[tuple[int, str]]:
    """
    從 meta.json 的 'ohlc_mapping' 取出 (pixel_x, date)
    回傳 list[(x_px, date_str)]
    """
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "ohlc_mapping" in meta and isinstance(meta["ohlc_mapping"], list):
            records = []
            for rec in meta["ohlc_mapping"]:
                if "date" in rec and "open" in rec and "pixel_x" in rec["open"]:
                    records.append((rec["open"]["pixel_x"], rec["date"]))
            return records
    except Exception as e:
        print(f"⚠️ load_date_list error: {e}")
    return []


def px_to_nearest_date(x_px: float, mapping: list[tuple[int, str]]) -> str:
    """
    給一個 x_px (像素) 與 (pixel_x, date) 對應列表，
    回傳最近日期字串
    """
    if not mapping:
        return "unknown"
    xs = np.array([m[0] for m in mapping], dtype=float)
    dates = [m[1] for m in mapping]
    idx = int(np.argmin(np.abs(xs - x_px)))
    return dates[idx]


## Triangle fitting utils

def order_quad_clockwise(quad: np.ndarray) -> Optional[np.ndarray]:
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:,1]-c[1], quad[:,0]-c[0])
    idx = np.argsort(ang)
    q = quad[idx].astype(np.float32)
    start = np.argmin(q[:,1])
    return np.roll(q, -start, axis=0)

def pick_quad_from_mask(mask_xy: np.ndarray) -> Optional[np.ndarray]:
    """以最小外接矩形找 segment 四角，根據座標分群固定輸出 (LT, RT, RB, LB)。"""
    if mask_xy is None or len(mask_xy) < 4:
        return None

    pts = np.array(mask_xy, dtype=np.float32).reshape(-1, 1, 2)

    # --- 凸包與最小外接矩形 ---
    hull = cv2.convexHull(pts).reshape(-1, 2)
    rect = cv2.minAreaRect(hull.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect)
    quad = order_quad_clockwise(box)

    # --- 四角吸附邊界最近點 ---
    snapped = []
    for p in quad:
        j = np.argmin(np.linalg.norm(hull - p, axis=1))
        snapped.append(hull[j])
    quad = np.array(snapped, dtype=np.float32)

    # --- 座標分類：先比 X 再比 Y ---
    xs = quad[:, 0]
    median_x = np.median(xs)

    left_pts = quad[quad[:, 0] <= median_x]
    right_pts = quad[quad[:, 0] > median_x]

    # 左邊上下
    lt = left_pts[np.argmin(left_pts[:, 1])]   # y 最小 → 上
    lb = left_pts[np.argmax(left_pts[:, 1])]   # y 最大 → 下

    # 右邊上下
    rt = right_pts[np.argmin(right_pts[:, 1])]
    rb = right_pts[np.argmax(right_pts[:, 1])]

    return np.array([lt, rt, rb, lb], dtype=np.float32)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def build_fixed_color_map(label_map):
    """固定 label→顏色映射，避免每張圖顏色飄移。"""
    cmap = plt.get_cmap("tab20")  # 20 色循環
    color_map = {}
    for i, lab in enumerate(label_map):
        lab = str(lab).strip()
        rgb = cmap(i % 20)[:3]
        bgr = tuple(int(255 * c) for c in rgb[::-1])
        color_map[lab] = bgr
    return color_map

def draw_instances_on_image(img_bgr, masks_xy, labels=None,
                            color_map=None, line_thickness=0,
                            fill_alpha=0.25, font_scale=0.5):
    """在原圖上畫多邊形與小字 label；使用固定 color_map。"""
    H, W = img_bgr.shape[:2]
    t = line_thickness if line_thickness > 0 else max(2, round(min(H, W) / 640 * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if masks_xy is None or len(masks_xy) == 0:
        return img_bgr

    overlay = img_bgr.copy()
    for idx, poly in enumerate(masks_xy):
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        label = (labels[idx] if labels and idx < len(labels) else "obj")
        label = str(label).strip()
        color = (0, 255, 0) if color_map is None else color_map.get(label, (0, 255, 0))

        if fill_alpha > 0:
            cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=t, lineType=cv2.LINE_AA)

        x, y = pts[0, 0]
        # cv2.putText(img_bgr, label, (x + 3, max(12, y - 5)),
        #             font, font_scale, color, 1, cv2.LINE_AA)

    if fill_alpha > 0:
        img_bgr = cv2.addWeighted(overlay, fill_alpha, img_bgr, 1 - fill_alpha, 0)
    return img_bgr

def count_inputs(source_dir):
    """估算輸入張數，用於 tqdm total。影片來源將被低估，但不影響運作。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    n = 0
    if os.path.isfile(source_dir):
        return 1
    for root, _, files in os.walk(source_dir):
        n += sum(1 for fn in files if os.path.splitext(fn)[1].lower() in exts)
    return n



def run_inference(model_path, source_dir, save_dir, output_dir,
                  model=None, max_pts=64, approx_eps=2.0, predict_imgsz=896,
                  fill_alpha=0.25, line_thickness=0, jpeg_quality=95):
    """
    predict_imgsz: 推論內部解析度（32 的倍數）。訓練 640 也可在此用 768/896/1024。
    """
    save_dir = os.path.abspath(save_dir)

    if model is None:
        model = YOLO(model_path)


    # 取得 label_map
    try:
        names = model.model.names if hasattr(model, "model") else model.names
        if isinstance(names, dict) and len(names) > 0:
            max_idx = max(names.keys())
            label_map = [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
        elif isinstance(names, list) and len(names) > 0:
            label_map = names
        else:
            label_map = label_names
    except Exception:
        label_map = label_names
    label_map = [str(x).strip() for x in label_map]


    # 固定色表（建一次，整批共用）
    fixed_color_map = build_fixed_color_map(label_map)

    # 推論
    results = model.predict(
        source=source_dir,
        imgsz=int(predict_imgsz),
        save=False,
        stream=True,
        verbose=False
    )

    output_records = []
    total = count_inputs(source_dir)
    
    for r in tqdm(results, total=total, unit="img", desc="Infer"):

        file_name = os.path.basename(getattr(r, "path", f"frame.png"))
        # 原圖
        img_path = getattr(r, "path", None)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"⚠️ Cannot read image: {img_path}")
            continue

        meta_path = img_path.replace("/image/", "/json/").replace(".png", ".json")
        with open(meta_path) as f:
            meta = json.load(f)

        date_list = load_date_list(meta_path)

        panel_top = meta["axes_map"]["price"]["panel_top"]
        panel_height = meta["axes_map"]["price"]["panel_height"]
        min_price = meta["axes_map"]["price"]["domain"]["min_price"]
        max_price = meta["axes_map"]["price"]["domain"]["max_price"]
        margin_top, margin_bottom = meta["axes_map"]["price"]["margin_top"], meta["axes_map"]["price"]["margin_bottom"]

        def px_to_price_norm(y_px):
            usable_h = panel_height - (margin_top + margin_bottom)
            norm_val = (panel_top + margin_top + usable_h - y_px) / usable_h
            return min_price + norm_val * (max_price - min_price)


        # 無 mask
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            out_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")
            cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            continue

        num_obj = len(r.masks.data)

        # 依 boxes 取每個實例的 label
        if r.boxes is not None and len(r.boxes) == num_obj:
            labels = [label_map[int(b.cls.item())].strip() for b in r.boxes]
        else:
            # 退化處理：無 boxes 時以第一類別名填補
            labels = [label_map[0]] * num_obj

        # 畫圖（固定色表）
        if SHOW_SEGMENT:
            img_drawn = draw_instances_on_image(
                img_bgr=img.copy(),
                masks_xy=r.masks.xy,
                labels=labels,
                color_map=fixed_color_map,
                line_thickness=line_thickness,
                fill_alpha=fill_alpha,
                font_scale=0.5
            )
        else:
            img_drawn = img.copy()



        # ===== 新增：畫 bounding box =====
        if SHOW_BOX and r.boxes is not None and len(r.boxes) > 0:
            for idx, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls.item()) if box.cls is not None else 0
                conf = float(box.conf.item()) if box.conf is not None else 0
                label_en = label_map[cls_id]

                # 中文轉換表
                label_rename = {
                    "Up-Triangle": "上三角",
                    "Down-Triangle": "下三角",
                    "Up-W": "W底",
                    "Down-M": "M頭",
                    "Up-Head-Shoulder-Bottom": "頭肩底",
                    "Down-Head-Shoulder-Top": "頭肩頂",
                    "Up-Breakout": "破底翻",
                    "Down-Breakout": "假突破",
                }
                label_zh = label_rename.get(label_en, label_en)
                label_text = f"{label_zh} {conf:.2f}"

                # 顏色依 Up/Down
                if label_en.startswith("Up-"):
                    color = (0, 0, 255)
                elif label_en.startswith("Down-"):
                    color = (0, 180, 0)
                else:
                    color = (0, 0, 0)

                cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(img_drawn, label_text, (x1, y1 - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                img_drawn = put_chinese_text(img_drawn, label_text, (x1, y1), color=color)

        elif not SHOW_BOX and r.boxes is not None and len(r.boxes) > 0:
            for idx, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls.item()) if box.cls is not None else 0
                conf = float(box.conf.item()) if box.conf is not None else 0
                label_en = label_map[cls_id]

                # 中文轉換表
                label_rename = {
                    "Up-Triangle": "上三角",
                    "Down-Triangle": "下三角",
                    "Up-W": "W底",
                    "Down-M": "M頭",
                    "Up-Head-Shoulder-Bottom": "頭肩底",
                    "Down-Head-Shoulder-Top": "頭肩頂",
                    "Up-Breakout": "破底翻",
                    "Down-Breakout": "假突破",
                }
                label_zh = label_rename.get(label_en, label_en)
                label_text = f"{label_zh} {conf:.2f}"

                # 顏色依 Up/Down
                if label_en.startswith("Up-"):
                    color = (0, 0, 255)
                elif label_en.startswith("Down-"):
                    color = (0, 180, 0)
                else:
                    color = (0, 0, 0)

                # cv2.putText(img_drawn, label_text, (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                img_drawn = put_chinese_text(img_drawn, label_text, (x1, y1*0.95), color=color)

        # =================================


        # === 根據類型畫出幾何結構 ===
        for idx in range(num_obj):
            try:
                xy = np.array(r.masks.xy[idx])
                if xy.shape[0] < 4:
                    continue
            except Exception:
                continue

            def record(label, date_ref, price_ref, price_1st, price_2nd):
                ret_1st = (price_1st - price_ref) / price_ref * 100.0
                ret_2nd = (price_2nd - price_ref) / price_ref * 100.0
                output_records.append({
                    "file": file_name, "label": label, "date_ref": date_ref,
                    "price_ref": round(price_ref, 2),
                    "price_1st": round(price_1st, 2), "ret_1st": round(ret_1st, 1),
                    "price_2nd": round(price_2nd, 2), "ret_2nd": round(ret_2nd, 1)
                })

            label = labels[idx]


            # === Up-Triangle ===
            if "Up-Triangle" in label:
                try:
                    quad = pick_quad_from_mask(xy)
                except Exception:
                    quad = None

                if quad is not None:
                    lt, rt, rb, lb = quad
                    # 高度差
                    h = lb[1] - lt[1]
                    x_ref, y_ref = int(rt[0]), int(rt[1])

                    price_ref = px_to_price_norm(y_ref)
                    date_ref = px_to_nearest_date(x_ref, date_list)


                    y_1st = int(y_ref - h)
                    y_2nd = int(y_ref - 2 * h)

                    # 黑色虛線框
                    for i in range(4):
                        p1 = tuple(np.int32(quad[i]))
                        p2 = tuple(np.int32(quad[(i + 1) % 4]))
                        # 畫成虛線（每隔 6px 畫 3px）
                        line_len = int(np.linalg.norm(np.array(p2) - np.array(p1)))
                        for t in range(0, line_len, 6):
                            pt_a = (
                                int(p1[0] + (p2[0] - p1[0]) * t / line_len),
                                int(p1[1] + (p2[1] - p1[1]) * t / line_len),
                            )
                            pt_b = (
                                int(p1[0] + (p2[0] - p1[0]) * (t + 3) / line_len),
                                int(p1[1] + (p2[1] - p1[1]) * (t + 3) / line_len),
                            )
                            cv2.line(img_drawn, pt_a, pt_b, (0, 0, 0), 2)

                    # 藍色四角點
                    for (x, y) in quad:
                        cv2.circle(img_drawn, (int(x), int(y)), 4, (255, 0, 0), -1)



                    # 垂直虛線
                    for y in range(min(y_ref, y_2nd) - 10, max(y_ref, y_2nd) + 10, 6):
                        if (y // 6) % 2 == 0:
                            cv2.line(img_drawn, (x_ref, y), (x_ref, y + 3), (0, 0, 0), 1)


                    # === 滿足點與報酬率輸出 ===
                    price_1st = px_to_price_norm(y_1st)
                    price_2nd = px_to_price_norm(y_2nd)
                    ret_1st = (price_1st - price_ref) / price_ref * 100.0
                    ret_2nd = (price_2nd - price_ref) / price_ref * 100.0

                    for y_target, tag, p_val, r_pct in [
                        (y_1st, "1st", price_1st, ret_1st),
                        (y_2nd, "2nd", price_2nd, ret_2nd)
                    ]:
                        cv2.circle(img_drawn, (x_ref, y_target), 3, (0, 0, 0), -1)
                        cv2.putText(
                            img_drawn,
                            f"{p_val:.2f} ({r_pct:+.1f}%)",
                            (x_ref + 5, y_target + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )
                    
                    record(label, date_ref, price_ref, price_1st, price_2nd)




            elif "Down-Triangle" in label:
                try:
                    quad = pick_quad_from_mask(xy)
                except Exception:
                    quad = None

                if quad is not None:
                    lt, rt, rb, lb = quad
                    h = lb[1] - lt[1]   # 高度差
                    x_ref, y_ref = int(rb[0]), int(rb[1])  # 右下角作為突破點
                    price_ref = px_to_price_norm(y_ref)
                    date_ref = px_to_nearest_date(x_ref, date_list)

                    # === 滿足點往下 ===
                    y_1st = int(y_ref + h)
                    y_2nd = int(y_ref + 2 * h)


                    # --- 黑色虛線框 ---
                    for i in range(4):
                        p1 = tuple(np.int32(quad[i]))
                        p2 = tuple(np.int32(quad[(i + 1) % 4]))
                        line_len = int(np.linalg.norm(np.array(p2) - np.array(p1)))
                        for t in range(0, line_len, 6):
                            pt_a = (
                                int(p1[0] + (p2[0] - p1[0]) * t / line_len),
                                int(p1[1] + (p2[1] - p1[1]) * t / line_len),
                            )
                            pt_b = (
                                int(p1[0] + (p2[0] - p1[0]) * (t + 3) / line_len),
                                int(p1[1] + (p2[1] - p1[1]) * (t + 3) / line_len),
                            )
                            cv2.line(img_drawn, pt_a, pt_b, (0, 0, 0), 2)

                    # --- 藍色角點 ---
                    for (x, y) in quad:
                        cv2.circle(img_drawn, (int(x), int(y)), 4, (255, 0, 0), -1)



                    # --- 黑色垂直虛線：從 RB 往下 ---
                    for y in range(y_ref, y_2nd + 10, 6):
                        if (y // 6) % 2 == 0:
                            cv2.line(img_drawn, (x_ref, y), (x_ref, y + 3), (0, 0, 0), 1)

                    # === 滿足點與報酬率輸出 ===
                    price_1st = px_to_price_norm(y_1st)
                    price_2nd = px_to_price_norm(y_2nd)
                    ret_1st = (price_1st - price_ref) / price_ref * 100.0
                    ret_2nd = (price_2nd - price_ref) / price_ref * 100.0

                    for y_target, tag, p_val, r_pct in [
                        (y_1st, "1st", price_1st, ret_1st),
                        (y_2nd, "2nd", price_2nd, ret_2nd)
                    ]:
                        cv2.circle(img_drawn, (x_ref, y_target), 3, (0, 0, 0), -1)
                        cv2.putText(
                            img_drawn,
                            f"{p_val:.2f} ({r_pct:+.1f}%)",
                            (x_ref + 5, y_target + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )
                    
                    record(label, date_ref, price_ref, price_1st, price_2nd)






            # === W 底 ===
            elif "Up-W" in label:
                try:
                    yolo_box = r.boxes[idx].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, yolo_box)
                    # xy = np.array(r.masks.xy[idx])
                    # xy = xy[np.argsort(xy[:, 0])]

                    # # 取左右最低點（谷底）
                    # left_bottom = xy[np.argmax(xy[:len(xy)//2, 1])]
                    # right_bottom = xy[len(xy)//2 + np.argmax(xy[len(xy)//2:, 1])]

                    xy = np.array(r.masks.xy[idx])
                    xy = xy[np.argsort(xy[:, 0])]  # 先依 x 排序
                    x, y = xy[:, 0], xy[:, 1]

                    segment_ratio = 0.4
                    x_left_max = x.min() + (x.max() - x.min()) * segment_ratio
                    x_right_min = x.max() - (x.max() - x.min()) * segment_ratio

                    left_pts = xy[x <= x_left_max]
                    right_pts = xy[x >= x_right_min]

                    # 左右谷（y 最大 = 最低點）
                    if len(left_pts) > 0:
                        left_bottom = left_pts[np.argmax(left_pts[:, 1])]
                    else:
                        left_bottom = xy[np.argmax(y)]

                    if len(right_pts) > 0:
                        right_bottom = right_pts[np.argmax(right_pts[:, 1])]
                    else:
                        right_bottom = xy[np.argmax(y)]



                    # Δh = 頂線 - 兩谷平均
                    avg_bottom_y = (left_bottom[1] + right_bottom[1]) / 2.0
                    delta_h = abs(y1 - avg_bottom_y)

                    # 起始點：右上角 (RT)
                    x_ref, y_ref = int(x2), int(y1)
                    y_1st = int(y_ref - delta_h)
                    y_2nd = int(y_ref - 2 * delta_h)

                    price_ref = px_to_price_norm(y_ref)
                    date_ref = px_to_nearest_date(x_ref, date_list)

                    # --- 灰色頂線 ---
                    cv2.line(img_drawn, (int(x1), int(y1)), (int(x2), int(y1)),
                            (180, 180, 180), 1, cv2.LINE_AA)

                    # --- 找中心點 (兩谷之間) ---
                    x_min = min(left_bottom[0], right_bottom[0])
                    x_max = max(left_bottom[0], right_bottom[0])
                    mid_mask = (xy[:, 0] > x_min) & (xy[:, 0] < x_max)

                    if np.any(mid_mask):
                        mid_xy = xy[mid_mask]
                        idx_center = np.argmin(np.abs(mid_xy[:, 1] - y1))
                        seg_center_pt = mid_xy[idx_center]
                    else:
                        idx_center = np.argmin(np.abs(xy[:, 1] - y1))
                        seg_center_pt = xy[idx_center]

                    # --- 投影到水平線 ---
                    center_proj_pt = np.array([seg_center_pt[0], y1], dtype=np.float32)

                    # --- 左右端點 ---
                    left_top_pt = np.array([x1, y1], dtype=np.float32)
                    right_top_pt = np.array([x2, y1], dtype=np.float32)

                    # --- 黑色虛線畫法 ---
                    def draw_dashed_line(img, pt1, pt2, color=(0,0,0), thickness=2, dash_len=6):
                        pt1, pt2 = np.int32(pt1), np.int32(pt2)
                        dist = int(np.hypot(*(pt2 - pt1)))
                        for i in range(0, dist, dash_len*2):
                            start = pt1 + (pt2 - pt1) * (i / dist)
                            end = pt1 + (pt2 - pt1) * ((i + dash_len) / dist)
                            cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), color, thickness, cv2.LINE_AA)

                    # --- 畫線（黑色虛線）---
                    draw_dashed_line(img_drawn, left_bottom, center_proj_pt)
                    draw_dashed_line(img_drawn, right_bottom, center_proj_pt)
                    draw_dashed_line(img_drawn, left_bottom, left_top_pt)
                    draw_dashed_line(img_drawn, right_bottom, right_top_pt)

                    # --- 藍色點：谷底、頂線端點、中心投影 ---
                    for p in [left_bottom, right_bottom, left_top_pt, right_top_pt, center_proj_pt]:
                        cv2.circle(img_drawn, tuple(np.int32(p)), 5, (255, 0, 0), -1)

    
                    # 黑色虛線 (往上)
                    for y in range(y_ref, y_2nd - 10, -6):
                        if (y // 6) % 2 == 0:
                            cv2.line(img_drawn, (x_ref, y), (x_ref, y - 3), (0, 0, 0), 1)

                    # # 標滿足點
                    # for y_target, tag in [(y_1st, "1st"), (y_2nd, "2nd")]:
                    #     price_val = px_to_price_norm(y_target)
                    #     ret_pct = (price_val - price_ref) / price_ref * 100.0
                    #     cv2.circle(img_drawn, (x_ref, y_target), 3, (0,0,0), -1)
                    #     cv2.putText(img_drawn, f"{price_val:.2f} ({ret_pct:+.1f}%)",
                    #                 (x_ref + 5, y_target + 5),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)


                    # === 滿足點與報酬率輸出 ===
                    price_1st = px_to_price_norm(y_1st)
                    price_2nd = px_to_price_norm(y_2nd)
                    ret_1st = (price_1st - price_ref) / price_ref * 100.0
                    ret_2nd = (price_2nd - price_ref) / price_ref * 100.0

                    for y_target, tag, p_val, r_pct in [
                        (y_1st, "1st", price_1st, ret_1st),
                        (y_2nd, "2nd", price_2nd, ret_2nd)
                    ]:
                        cv2.circle(img_drawn, (x_ref, y_target), 3, (0, 0, 0), -1)
                        cv2.putText(
                            img_drawn,
                            f"{p_val:.2f} ({r_pct:+.1f}%)",
                            (x_ref + 5, y_target + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )
                    
                    record(label, date_ref, price_ref, price_1st, price_2nd)


                except Exception as e:
                    tqdm.write(f"[warn] Up-W analysis failed for {file_name}: {e}")


            # === M 頭 ===
            elif "Down-M" in label:
                try:
                    yolo_box = r.boxes[idx].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, yolo_box)

                    # xy = np.array(r.masks.xy[idx])
                    # xy = xy[np.argsort(xy[:, 0])]

                    # # 取左右最高點（峰頂）
                    # left_top = xy[np.argmin(xy[:len(xy)//2, 1])]
                    # right_top = xy[len(xy)//2 + np.argmin(xy[len(xy)//2:, 1])]


                    xy = np.array(r.masks.xy[idx])
                    xy = xy[np.argsort(xy[:, 0])]  # 先依 x 排序
                    x, y = xy[:, 0], xy[:, 1]

                    # --- 左右邊界範圍比例（可微調）---
                    segment_ratio = 0.4
                    x_left_max = x.min() + (x.max() - x.min()) * segment_ratio
                    x_right_min = x.max() - (x.max() - x.min()) * segment_ratio

                    # --- 左右分段 ---
                    left_pts = xy[x <= x_left_max]
                    right_pts = xy[x >= x_right_min]

                    # --- 左側峰：y 最小（最高點）---
                    if len(left_pts) > 0:
                        left_top = left_pts[np.argmin(left_pts[:, 1])]
                    else:
                        left_top = xy[np.argmin(y)]

                    # --- 右側峰：y 最小（最高點）---
                    if len(right_pts) > 0:
                        right_top = right_pts[np.argmin(right_pts[:, 1])]
                    else:
                        right_top = xy[np.argmin(y)]

                    # Δh = 兩峰平均 - 底線
                    avg_top_y = (left_top[1] + right_top[1]) / 2.0
                    delta_h = abs(avg_top_y - y2)

                    # 起始點：右下角 (RB)
                    x_ref, y_ref = int(x2), int(y2)
                    y_1st = int(y_ref + delta_h)
                    y_2nd = int(y_ref + 2 * delta_h)
                    price_ref = px_to_price_norm(y_ref)
                    date_ref = px_to_nearest_date(x_ref, date_list)

                    # --- 灰色底線 ---
                    cv2.line(img_drawn, (int(x1), int(y2)), (int(x2), int(y2)),
                            (180, 180, 180), 1, cv2.LINE_AA)

                    # --- 找中心點 (兩峰之間) ---
                    x_min = min(left_top[0], right_top[0])
                    x_max = max(left_top[0], right_top[0])
                    mid_mask = (xy[:, 0] > x_min) & (xy[:, 0] < x_max)

                    if np.any(mid_mask):
                        mid_xy = xy[mid_mask]
                        idx_center = np.argmin(np.abs(mid_xy[:, 1] - y2))
                        seg_center_pt = mid_xy[idx_center]
                    else:
                        idx_center = np.argmin(np.abs(xy[:, 1] - y2))
                        seg_center_pt = xy[idx_center]

                    # --- 投影到底線 ---
                    center_proj_pt = np.array([seg_center_pt[0], y2], dtype=np.float32)

                    # --- 左右端點 ---
                    left_bot_pt = np.array([x1, y2], dtype=np.float32)
                    right_bot_pt = np.array([x2, y2], dtype=np.float32)

                    # --- 黑色虛線畫法 ---
                    def draw_dashed_line(img, pt1, pt2, color=(0,0,0), thickness=2, dash_len=6):
                        pt1, pt2 = np.int32(pt1), np.int32(pt2)
                        dist = int(np.hypot(*(pt2 - pt1)))
                        for i in range(0, dist, dash_len*2):
                            start = pt1 + (pt2 - pt1) * (i / dist)
                            end = pt1 + (pt2 - pt1) * ((i + dash_len) / dist)
                            cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), color, thickness, cv2.LINE_AA)

                    # --- 畫線（黑色虛線）---
                    draw_dashed_line(img_drawn, left_top, center_proj_pt)
                    draw_dashed_line(img_drawn, right_top, center_proj_pt)
                    draw_dashed_line(img_drawn, left_top, left_bot_pt)
                    draw_dashed_line(img_drawn, right_top, right_bot_pt)

                    # --- 藍色點：兩峰、底線端點、中心投影 ---
                    for p in [left_top, right_top, left_bot_pt, right_bot_pt, center_proj_pt]:
                        cv2.circle(img_drawn, tuple(np.int32(p)), 5, (255, 0, 0), -1)

                    # 黑色虛線 (往下)
                    for y in range(y_ref, y_2nd + 10, 6):
                        if (y // 6) % 2 == 0:
                            cv2.line(img_drawn, (x_ref, y), (x_ref, y + 3), (0, 0, 0), 1)

                    # # 標滿足點
                    # for y_target, tag in [(y_1st, "1st"), (y_2nd, "2nd")]:
                    #     price_val = px_to_price_norm(y_target)
                    #     ret_pct = (price_val - price_ref) / price_ref * 100.0
                    #     cv2.circle(img_drawn, (x_ref, y_target), 3, (0,0,0), -1)
                    #     cv2.putText(img_drawn, f"{price_val:.2f} ({ret_pct:+.1f}%)",
                    #                 (x_ref + 5, y_target + 5),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)


                    # === 滿足點與報酬率輸出 ===
                    price_1st = px_to_price_norm(y_1st)
                    price_2nd = px_to_price_norm(y_2nd)
                    ret_1st = (price_1st - price_ref) / price_ref * 100.0
                    ret_2nd = (price_2nd - price_ref) / price_ref * 100.0

                    for y_target, tag, p_val, r_pct in [
                        (y_1st, "1st", price_1st, ret_1st),
                        (y_2nd, "2nd", price_2nd, ret_2nd)
                    ]:
                        cv2.circle(img_drawn, (x_ref, y_target), 3, (0, 0, 0), -1)
                        cv2.putText(
                            img_drawn,
                            f"{p_val:.2f} ({r_pct:+.1f}%)",
                            (x_ref + 5, y_target + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )
                    
                    record(label, date_ref, price_ref, price_1st, price_2nd)



                except Exception as e:
                    tqdm.write(f"[warn] Down-M analysis failed for {file_name}: {e}")


            elif "Down-Head-Shoulder-Top" in label:
                try:
                    yolo_box = r.boxes[idx].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, yolo_box)
                    xy = np.array(r.masks.xy[idx])
                    xy = xy[np.argsort(xy[:, 0])]  # 先依 X 排序
                    x, y = xy[:, 0], xy[:, 1]

                    # === 以左右分區比例找三個峰 ===
                    segment_ratio = 0.3
                    x_left_max = x.min() + (x.max() - x.min()) * segment_ratio
                    x_right_min = x.max() - (x.max() - x.min()) * segment_ratio

                    left_pts = xy[x <= x_left_max]
                    right_pts = xy[x >= x_right_min]

                    # 左峰（最高點 y 最小）
                    if len(left_pts) > 0:
                        left_peak = left_pts[np.argmin(left_pts[:, 1])]
                    else:
                        left_peak = xy[np.argmin(y)]

                    # 右峰（最高點 y 最小）
                    if len(right_pts) > 0:
                        right_peak = right_pts[np.argmin(right_pts[:, 1])]
                    else:
                        right_peak = xy[np.argmin(y)]

                    # 中間峰：在左右兩峰之間找 y 最小的點
                    x_min, x_max = min(left_peak[0], right_peak[0]), max(left_peak[0], right_peak[0])
                    mid_region = xy[(xy[:, 0] > x_min) & (xy[:, 0] < x_max)]
                    if len(mid_region) > 0:
                        mid_peak = mid_region[np.argmin(mid_region[:, 1])]
                    else:
                        mid_peak = xy[np.argmin(y)]

                    peaks = np.vstack([left_peak, mid_peak, right_peak])

                    # === 底線 ===
                    base_line = np.array([[x1, y2], [x2, y2]], dtype=np.float32)
                    left_base, right_base = base_line

                    # === 找三個峰的投影點（在底線上） ===
                    proj_pts = np.array([[p[0], y2] for p in peaks], dtype=np.float32)

                    # --- 虛線繪製函式 ---
                    def draw_dashed_line(img, pt1, pt2, color=(0,0,0), thickness=2, dash_len=6):
                        pt1, pt2 = np.int32(pt1), np.int32(pt2)
                        dist = int(np.hypot(*(pt2 - pt1)))
                        for i in range(0, dist, dash_len*2):
                            start = pt1 + (pt2 - pt1) * (i / dist)
                            end = pt1 + (pt2 - pt1) * ((i + dash_len) / dist)
                            cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), color, thickness, cv2.LINE_AA)

                    # --- 畫虛線連線 ---
                    x_positions = [x1, 
                                (left_peak[0] + mid_peak[0]) / 2,
                                (mid_peak[0] + right_peak[0]) / 2,
                                x2]
                    base_pts = np.array([[xv, y2] for xv in x_positions], dtype=np.float32)

                    # --- 虛線連線：左底→左肩→頭→右肩→右底 ---
                    draw_dashed_line(img_drawn, base_pts[0], left_peak)
                    draw_dashed_line(img_drawn, left_peak, (x_positions[1], y2))
                    draw_dashed_line(img_drawn, (x_positions[1], y2), mid_peak)
                    draw_dashed_line(img_drawn, mid_peak, (x_positions[2], y2))
                    draw_dashed_line(img_drawn, (x_positions[2], y2), right_peak)
                    draw_dashed_line(img_drawn, right_peak, base_pts[-1])


                    # --- 藍點 ---
                    for p in [left_peak, mid_peak, right_peak]:
                        cv2.circle(img_drawn, tuple(np.int32(p)), 5, (255,0,0), -1)
                    for p in base_pts:
                        cv2.circle(img_drawn, tuple(np.int32(p)), 4, (255,0,0), -1)

                    # === 高度差與報酬率 ===
                    highest_peak = peaks[np.argmin(peaks[:, 1])]
                    delta_h = abs(highest_peak[1] - y2)
                    x_ref, y_ref = int(x2), int(y2)
                    y_1st = int(y_ref + delta_h)
                    y_2nd = int(y_ref + 2 * delta_h)
                    price_ref = px_to_price_norm(y_ref)
                    date_ref = px_to_nearest_date(x_ref, date_list)

                    # 黑色垂直虛線往下
                    for y_line in range(y_ref, y_2nd + 10, 6):
                        if (y_line // 6) % 2 == 0:
                            cv2.line(img_drawn, (x_ref, y_line), (x_ref, y_line + 3), (0, 0, 0), 1)

                    # # 標滿足點與報酬
                    # for y_target in [y_1st, y_2nd]:
                    #     price_val = px_to_price_norm(y_target)
                    #     ret_pct = (price_val - price_ref) / price_ref * 100.0
                    #     cv2.circle(img_drawn, (x_ref, y_target), 3, (0,0,0), -1)
                    #     cv2.putText(
                    #         img_drawn, f"{price_val:.2f} ({ret_pct:+.1f}%)",
                    #         (x_ref + 5, y_target + 5),
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA
                    #     )


                    # === 滿足點與報酬率輸出 ===
                    price_1st = px_to_price_norm(y_1st)
                    price_2nd = px_to_price_norm(y_2nd)
                    ret_1st = (price_1st - price_ref) / price_ref * 100.0
                    ret_2nd = (price_2nd - price_ref) / price_ref * 100.0

                    for y_target, tag, p_val, r_pct in [
                        (y_1st, "1st", price_1st, ret_1st),
                        (y_2nd, "2nd", price_2nd, ret_2nd)
                    ]:
                        cv2.circle(img_drawn, (x_ref, y_target), 3, (0, 0, 0), -1)
                        cv2.putText(
                            img_drawn,
                            f"{p_val:.2f} ({r_pct:+.1f}%)",
                            (x_ref + 5, y_target + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )
                    
                    record(label, date_ref, price_ref, price_1st, price_2nd)


                except Exception as e:
                    tqdm.write(f"[warn] Head-Shoulder Top failed for {file_name}: {e}")


            elif "Up-Head-Shoulder-Bottom" in label:
                try:
                    yolo_box = r.boxes[idx].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, yolo_box)
                    xy = np.array(r.masks.xy[idx])
                    xy = xy[np.argsort(xy[:, 0])]  # 依 X 排序
                    x, y = xy[:, 0], xy[:, 1]

                    # === 左右區分比例 ===
                    segment_ratio = 0.3
                    x_left_max = x.min() + (x.max() - x.min()) * segment_ratio
                    x_right_min = x.max() - (x.max() - x.min()) * segment_ratio

                    left_pts = xy[x <= x_left_max]
                    right_pts = xy[x >= x_right_min]

                    # 左谷（最低 y 最大）
                    if len(left_pts) > 0:
                        left_val = left_pts[np.argmax(left_pts[:, 1])]
                    else:
                        left_val = xy[np.argmax(y)]

                    # 右谷（最低 y 最大）
                    if len(right_pts) > 0:
                        right_val = right_pts[np.argmax(right_pts[:, 1])]
                    else:
                        right_val = xy[np.argmax(y)]

                    # 中谷：在左右谷之間取 y 最大
                    x_min, x_max = min(left_val[0], right_val[0]), max(left_val[0], right_val[0])
                    mid_region = xy[(xy[:, 0] > x_min) & (xy[:, 0] < x_max)]
                    if len(mid_region) > 0:
                        mid_val = mid_region[np.argmax(mid_region[:, 1])]
                    else:
                        mid_val = xy[np.argmax(y)]

                    valleys = np.vstack([left_val, mid_val, right_val])

                    # === 頂線 ===
                    top_line = np.array([[x1, y1], [x2, y1]], dtype=np.float32)

                    # --- 虛線繪製函式 ---
                    def draw_dashed_line(img, pt1, pt2, color=(0,0,0), thickness=2, dash_len=6):
                        pt1, pt2 = np.int32(pt1), np.int32(pt2)
                        dist = int(np.hypot(*(pt2 - pt1)))
                        for i in range(0, dist, dash_len*2):
                            start = pt1 + (pt2 - pt1) * (i / dist)
                            end = pt1 + (pt2 - pt1) * ((i + dash_len) / dist)
                            cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), color, thickness, cv2.LINE_AA)

                    # === 底線四個 X 位置 ===
                    x_positions = [
                        x1,
                        (left_val[0] + mid_val[0]) / 2,
                        (mid_val[0] + right_val[0]) / 2,
                        x2
                    ]
                    base_pts = np.array([[xv, y1] for xv in x_positions], dtype=np.float32)

                    # === 虛線連線：上底→左谷→頂線中點→右谷→上底 ===
                    draw_dashed_line(img_drawn, base_pts[0], left_val)
                    draw_dashed_line(img_drawn, left_val, (x_positions[1], y1))
                    draw_dashed_line(img_drawn, (x_positions[1], y1), mid_val)
                    draw_dashed_line(img_drawn, mid_val, (x_positions[2], y1))
                    draw_dashed_line(img_drawn, (x_positions[2], y1), right_val)
                    draw_dashed_line(img_drawn, right_val, base_pts[-1])

                    # --- 藍點 ---
                    for p in [left_val, mid_val, right_val]:
                        cv2.circle(img_drawn, tuple(np.int32(p)), 5, (255,0,0), -1)
                    for p in base_pts:
                        cv2.circle(img_drawn, tuple(np.int32(p)), 4, (255,0,0), -1)

                    # === 高度差與報酬率 ===
                    lowest_val = valleys[np.argmax(valleys[:, 1])]
                    delta_h = abs(y1 - lowest_val[1])
                    x_ref, y_ref = int(x2), int(y1)
                    y_1st = int(y_ref - delta_h)
                    y_2nd = int(y_ref - 2 * delta_h)
                    price_ref = px_to_price_norm(y_ref)
                    date_ref = px_to_nearest_date(x_ref, date_list)

                    # 黑色垂直虛線往上
                    for y_line in range(y_ref, y_2nd - 10, -6):
                        if (y_line // 6) % 2 == 0:
                            cv2.line(img_drawn, (x_ref, y_line), (x_ref, y_line - 3), (0, 0, 0), 1)

                    # # 標滿足點與報酬
                    # for y_target in [y_1st, y_2nd]:
                    #     price_val = px_to_price_norm(y_target)
                    #     ret_pct = (price_val - price_ref) / price_ref * 100.0
                    #     cv2.circle(img_drawn, (x_ref, y_target), 3, (0,0,0), -1)
                    #     cv2.putText(
                    #         img_drawn, f"{price_val:.2f} ({ret_pct:+.1f}%)",
                    #         (x_ref + 5, y_target + 5),
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA
                    #     )


                    # === 滿足點與報酬率輸出 ===
                    price_1st = px_to_price_norm(y_1st)
                    price_2nd = px_to_price_norm(y_2nd)
                    ret_1st = (price_1st - price_ref) / price_ref * 100.0
                    ret_2nd = (price_2nd - price_ref) / price_ref * 100.0

                    for y_target, tag, p_val, r_pct in [
                        (y_1st, "1st", price_1st, ret_1st),
                        (y_2nd, "2nd", price_2nd, ret_2nd)
                    ]:
                        cv2.circle(img_drawn, (x_ref, y_target), 3, (0, 0, 0), -1)
                        cv2.putText(
                            img_drawn,
                            f"{p_val:.2f} ({r_pct:+.1f}%)",
                            (x_ref + 5, y_target + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA
                        )
                    
                    record(label, date_ref, price_ref, price_1st, price_2nd)


                except Exception as e:
                    tqdm.write(f"[warn] Head-Shoulder Bottom failed for {file_name}: {e}")

        out_png = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")
        cv2.imwrite(out_png, img_drawn, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    df = pd.DataFrame(output_records)
    print(f"✅ Inference done. {len(df)} detections. Images saved to {output_dir}")
    return df



class Alpha_v1(Strategy):
    """
    每天畫一張圖 + 直接呼叫 run_inference()。
    不輸出 CSV。當天右側突破才建立交易訊號。
    """
    plot_script = "/Users/meng-jutsai/Stock/FiveB/script/plot_from_sql.py"
    model_path = "/Users/meng-jutsai/Stock/FiveB/runs/segment/yolov11m_seg_003/weights/best.pt"
    stop_loss_ratio = 0.1
    right_edge_ratio = 0.85
    long_labels = {"Up-Triangle", "Up-W", "Up-Head-Shoulder-Bottom"}
    # long_labels = {"Up-Triangle"}

    
    short_labels = {"Down-Triangle", "Down-M", "Down-Head-Shoulder-Top"}

    max_add = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = global_time
        self.tmp_dir = f"/Users/meng-jutsai/Stock/FiveB/results/backtest/temp/"
        os.makedirs(self.tmp_dir, exist_ok=True)
        # === 預載 YOLO 模型（只載入一次） ===
        print(f"🧠 Loading YOLO model once: {self.model_path}")
        self.yolo_model = YOLO(self.model_path)

    def _plot_single(self, stock_id, start_date, end_date):
        """呼叫 plot_from_sql.py 繪出單張圖，回傳影像檔完整路徑"""

        # === 取得股票名稱 ===
        conn = sqlite3.connect("/Users/meng-jutsai/Stock/FiveB/stock.db")
        try:
            stock_name = pd.read_sql_query(
                "SELECT stock_name FROM tw_stock_info WHERE stock_id = ?",
                conn, params=(stock_id,)
            )["stock_name"].squeeze()
        except Exception:
            stock_name = ""
        finally:
            conn.close()

        stock_dir = os.path.join(self.tmp_dir, self.timestamp, f"{stock_id}_{stock_name}")
        image_dir = os.path.join(stock_dir, "image")
        os.makedirs(image_dir, exist_ok=True)

        # === 呼叫繪圖腳本 ===
        cmd = [
            "python", self.plot_script,
            "--stock_id", stock_id,
            "--start_date", "2000-01-01",
            "--end_date", end_date,
            "--freq", "D",
            "--output_dir", stock_dir,     # ← 改成以股票為單位
            "--date_folder", "skip"
        ]
        subprocess.run(cmd, check=False)

        # === 根據 plot_from_sql.py 的命名邏輯組檔名 ===
        fname_base = f"{stock_id}_{stock_name}_2000-01-01_{end_date}_D-K_raw"
        img_path = os.path.join(image_dir, f"{fname_base}_norm.png")

        # === fallback ===
        if not os.path.exists(img_path):
            image_files = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            img_path = max(image_files, key=os.path.getmtime) if image_files else None

        if img_path is None:
            print(f"[warn] 無法在 {image_dir} 找到影像")

        return img_path

    def create_trade_sign(self, stock_price, **kwargs):
        import os, sqlite3, numpy as np, pandas as pd
        stock_id = kwargs.get("stock_id", self.stock_id)
        conn = sqlite3.connect("/Users/meng-jutsai/Stock/FiveB/stock.db")

        sp = stock_price.copy()
        sp["date"] = pd.to_datetime(sp["date"])
        sp["signal"] = 0
        sp["position"] = 0
        sp["target1"] = np.nan
        sp["target2"] = np.nan
        sp["stop_price"] = np.nan
        sp["exit_price"] = np.nan
        sp["exit_reason"] = ""

        active_trades = []  # FIFO list [{entry_date, entry_price, target1, target2, stop_price, position}]
        records = []

        for i, row in sp.iterrows():
            date = row["date"]
            price = row["close"]
            high = row["max"]
            low = row["min"]

            # === 非交易日略過 ===
            q = pd.read_sql_query("SELECT is_trading FROM tw_trading_calendar WHERE date=?",
                                conn, params=(date.strftime("%Y-%m-%d"),))
            if q.empty or q.iloc[0, 0] == 0:
                sp.loc[i, "position"] = sum(t["position"] for t in active_trades)
                continue

            # === 先檢查持倉出場 ===
            to_remove = []
            for idx, trade in enumerate(active_trades):
                exit_qty = 0
                exit_price = np.nan
                exit_reason = None

                # 停損
                if low <= trade["stop_price"]:
                    exit_qty = trade["position"]
                    exit_price = trade["stop_price"]
                    exit_reason = "STOP"
                    to_remove.append(idx)

                # 第一滿足 → 出場
                elif high >= trade["target1"]:
                    exit_qty = trade["position"]
                    exit_price = trade["target1"]
                    exit_reason = "TP1"
                    to_remove.append(idx)

                # 第二滿足 → 出場（若第一滿足未觸發）
                elif high >= trade["target2"]:
                    exit_qty = trade["position"]
                    exit_price = trade["target2"]
                    exit_reason = "TP2"
                    to_remove.append(idx)

                if exit_qty > 0:
                    sp.loc[i, "signal"] -= exit_qty
                    sp.loc[i, "exit_price"] = exit_price
                    sp.loc[i, "exit_reason"] = exit_reason
                    records.append({
                        "date": date,
                        "action": exit_reason,
                        "price": exit_price,
                        "qty": -exit_qty,
                        "entry_date": trade["entry_date"],
                        "entry_price": trade["entry_price"]
                    })

            # FIFO 移除賣出的持倉
            for idx in sorted(to_remove, reverse=True):
                active_trades.pop(idx)

            # === 新偵測（多頭型態） ===
            img_path = self._plot_single(
                stock_id,
                (date - pd.Timedelta(days=360)).strftime("%Y-%m-%d"),
                date.strftime("%Y-%m-%d")
            )
            if not img_path or not os.path.exists(img_path):
                sp.loc[i, "position"] = sum(t["position"] for t in active_trades)
                continue

            output_dir = os.path.dirname(img_path.replace("/image/", "/image-seg/"))
            os.makedirs(output_dir, exist_ok=True)

            try:
                df = run_inference(self.model_path, img_path, self.tmp_dir, output_dir, model=self.yolo_model)

            except Exception as e:
                print(f"[warn] run_inference failed @ {date}: {e}")
                df = None

            # === 當日新訊號 ===
            df_today = df[df["date_ref"] == date.strftime("%Y-%m-%d")] if df is not None else pd.DataFrame()
            df_today = df_today[df_today["label"].isin(self.long_labels)]
            if not df_today.empty:
                # 當天取 price_1st 最大者
                best = df_today.loc[df_today["price_1st"].idxmax()]
                p1, p2 = best["price_1st"], best["price_2nd"]
                stop_tmp = price * (1 - self.stop_loss_ratio)
                rr = (p1 - price) / (price - stop_tmp + 1e-9)

                add_units = min(max(int(rr), 1), getattr(self, "max_add", 3))

                trade = dict(
                    entry_date=date,
                    entry_price=price,
                    target1=p1,
                    target2=p2,
                    stop_price=stop_tmp,
                    position=add_units
                )
                active_trades.append(trade)

                sp.loc[i, ["signal", "target1", "target2", "stop_price"]] = [add_units, p1, p2, stop_tmp]
                records.append({
                    "date": date, "action": "BUY", "price": price,
                    "target1": p1, "target2": p2, "stop": stop_tmp,
                    "qty": add_units, "rr": round(rr, 2)
                })

            # === 更新持倉 ===
            sp.loc[i, "position"] = sum(t["position"] for t in active_trades)

        conn.close()
        sp["date"] = sp["date"].dt.strftime("%Y-%m-%d")

        # === 儲存交易紀錄 ===

        trade_log_dir = os.path.join(self.tmp_dir, self.timestamp)

        out_log = f"{trade_log_dir}/results/backtest/trade_log_{stock_id}.csv"
        os.makedirs(os.path.dirname(out_log), exist_ok=True)
        pd.DataFrame(records).to_csv(out_log, index=False)
        print(f"💾 Trade log saved to {out_log}")

        return sp
