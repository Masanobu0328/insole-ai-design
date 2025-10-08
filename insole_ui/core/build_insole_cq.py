# core/build_insole_cq.py
import json
import os
import pandas as pd
import numpy as np
import cadquery as cq
from pathlib import Path

def _load_outline_points(csv_path: Path):
    df = pd.read_csv(csv_path)
    # 列名に x_mm,y_mm が来ても x,y に揃える
    cols = {c.lower(): c for c in df.columns}
    if "x" in cols and "y" in cols:
        xcol, ycol = cols["x"], cols["y"]
    elif "x_mm" in cols and "y_mm" in cols:
        xcol, ycol = cols["x_mm"], cols["y_mm"]
    else:
        raise ValueError("CSVには x,y または x_mm,y_mm の列が必要です")
    pts = list(zip(df[xcol].astype(float), df[ycol].astype(float)))
    if len(pts) < 5:
        raise ValueError(f"輪郭点が少なすぎます: {len(pts)}点（最低でも5点以上推奨）")
    # 自動で閉じる
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts

def _autoscale_to_length(pts, target_length_mm: float | None):
    """x方向の全長を target_length_mm に合わせてスケール（任意）"""
    if not target_length_mm:
        return pts
    xs = [p[0] for p in pts]
    cur_len = max(xs) - min(xs)
    if cur_len <= 0:
        return pts
    s = target_length_mm / cur_len
    return [(x * s, y * s) for (x, y) in pts]

def build_insole_from_csv(
    csv_path: Path,
    *,
    side: str = "right",               # ← 受け取れるようにしておく（未使用でもOK）
    thickness: float = 5.0,
    target_length_mm: float | None = None,
) -> cq.Workplane:
    """
    CSV輪郭から安全にインソールソリッドを作る（Loftでなく押し出し）。
    main.py から exporter で STL/STEP を出す前提なので、ここは solid を返す。
    """
    pts = _load_outline_points(csv_path)
    pts = _autoscale_to_length(pts, target_length_mm)
    # Workplane -> 面 -> 押し出し
    wp = cq.Workplane("XY").polyline(pts).close()
    solid = wp.extrude(thickness)
    return solid

def build_insole_from_json_csv(
    json_path: Path,
    csv_path: Path,
    side: str = "right",
    thickness: float = 5.0,
) -> cq.Workplane:
    """
    既存の main.py 互換のラッパー関数。
    JSON の foot.length があればそれに x方向全長を合わせる。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    target_len = None
    try:
        target_len = float(cfg.get("foot", {}).get("length", None))
    except Exception:
        target_len = None

    return build_insole_from_csv(
        csv_path=csv_path,
        side=side,
        thickness=thickness,
        target_length_mm=target_len,
    )
