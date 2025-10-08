# gait_analysis/preprocess.py
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional

# ---------- 基本ユーティリティ ----------
def interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="linear", limit_direction="both")

def trim_head(df: pd.DataFrame, ratio: float = 0.05) -> pd.DataFrame:
    if len(df) == 0: 
        return df
    head = int(max(0, min(len(df)//2, round(len(df)*ratio))))
    return df.iloc[head:].reset_index(drop=True)

def clip_by_sigma(s: pd.Series, sigma: float = 3.0) -> pd.Series:
    med = s.median()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s
    lower, upper = med - sigma*std, med + sigma*std
    return s.clip(lower=lower, upper=upper)

def clip_df_by_sigma(df: pd.DataFrame, sigma: float = 3.0, cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = cols or df.columns.tolist()
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = clip_by_sigma(df[c], sigma)
    return df

# ---------- スケール(cm/px) ----------
def estimate_scale_cm_per_px(df: pd.DataFrame, fallback: float = 0.2) -> float:
    """
    簡易推定：身長 ~ 170cm を仮定し、肩〜足首距離の中央値などから近似。
    データが乏しい場合は fallback を返す（例 0.2 cm/px → 5px=1cm）。
    """
    candidates = []
    def safe_dist(ax, ay, bx, by):
        return np.sqrt((df[ax]-df[bx])**2 + (df[ay]-df[by])**2)
    needed = {'left_shoulder_x','left_shoulder_y','right_ankle_x','right_ankle_y'}
    if all(c in df.columns for c in needed):
        d = safe_dist('left_shoulder_x','left_shoulder_y','right_ankle_x','right_ankle_y').median()
        if np.isfinite(d) and d > 0:
            candidates.append(170.0 / d)  # 170cm / px
    return float(np.median(candidates)) if candidates else fallback

def load_scale_or_estimate(df: pd.DataFrame, scale_json_path: Optional[str]) -> float:
    if scale_json_path:
        try:
            with open(scale_json_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            if "scale_cm_per_px" in j and j["scale_cm_per_px"] > 0:
                return float(j["scale_cm_per_px"])
        except Exception:
            pass
    return estimate_scale_cm_per_px(df)

# ---------- イベント検出・正規化 ----------
def detect_heel_strikes(heel_y: pd.Series, prominence: float = 1.0) -> np.ndarray:
    """
    ヒールストライク＝Yの局所最小（find_peaksは最大検出なので -Y に適用）
    """
    if heel_y.isnull().all():
        return np.array([], dtype=int)
    y = heel_y.fillna(heel_y.median()).values
    peaks, _ = find_peaks(-y, prominence=prominence)
    return peaks

def normalize_heel_y_by_step(df: pd.DataFrame, heel_y_col: str, strikes: np.ndarray) -> pd.Series:
    """
    区間（ストライク間）ごとに最小値を0へ平行移動。解釈しやすい“接地=0”の波形に。
    """
    y = df[heel_y_col].copy()
    if len(strikes) == 0:
        return y - y.min()
    out = y.copy()
    idxs = list(strikes) + [len(df)-1]
    prev = 0
    for s in idxs:
        seg = out.iloc[prev:s+1]
        out.iloc[prev:s+1] = seg - seg.min()
        prev = s+1
    return out

# ---------- 骨盤スウェイ（定義の正規化） ----------
def pelvis_center_x(df: pd.DataFrame) -> pd.Series:
    """
    腰中心X = 左右のウエスト/ASISに近いランドマーク中点。
    なければ left/right_hip_x の中点で代替。
    """
    if all(c in df.columns for c in ["left_hip_x","right_hip_x"]):
        return (df["left_hip_x"] + df["right_hip_x"]) / 2.0
    # フォールバック：肩の中点
    if all(c in df.columns for c in ["left_shoulder_x","right_shoulder_x"]):
        return (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2.0
    # 最終フォールバック：0
    return pd.Series(np.zeros(len(df)))

def pelvic_lateral_sway_cm(df: pd.DataFrame, scale_cm_per_px: float) -> pd.Series:
    # X方向の変位（横揺れ）。中央値基準で中心化 → cm換算。
    cx = pelvis_center_x(df)
    centered = cx - cx.median()
    return centered * scale_cm_per_px

# ---------- 相互相関でのラグ推定 ----------
def estimate_lag(a: pd.Series, b: pd.Series, max_lag: int = 60) -> int:
    """
    a と b の系列の相互相関最大点のラグ（b をシフトすべきフレーム数、+でbを遅らせる）。
    """
    x = a.fillna(a.median()).values
    y = b.fillna(b.median()).values
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    lags = range(-max_lag, max_lag+1)
    best_lag, best_corr = 0, -1e9
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x[-lag:], y[:n+lag])[0,1]
        elif lag > 0:
            corr = np.corrcoef(x[:n-lag], y[lag:])[0,1]
        else:
            corr = np.corrcoef(x, y)[0,1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag

def shift_df(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    if lag == 0:
        return df
    if lag > 0:
        # df を遅らせる（先頭にNaNを入れる）
        pad = pd.DataFrame({c:[np.nan]*lag for c in df.columns})
        out = pd.concat([pad, df], ignore_index=True)
        return out.iloc[:len(df)].reset_index(drop=True)
    else:
        # 進める（末尾カット）
        out = df.iloc[-lag:].reset_index(drop=True)
        pad = pd.DataFrame({c:[np.nan]*(-lag) for c in df.columns})
        out = pd.concat([out, pad], ignore_index=True)
        return out.iloc[:len(df)].reset_index(drop=True)
