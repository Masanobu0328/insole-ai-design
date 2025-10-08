# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # 画像保存専用（GUI不要環境で確実に保存）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import json
from gait_analysis.preprocess import (
    interpolate_df, trim_head, clip_df_by_sigma,
    load_scale_or_estimate, detect_heel_strikes, normalize_heel_y_by_step,
    pelvic_lateral_sway_cm
)

# ====== 基本設定 ======
FPS = 30
SMOOTH_WINDOW = 5
MIN_STEP_DISTANCE = 8
PERCENTILES_TRY = [5,10,15,20,25,30,35,40,45]
PNGSAVE_KW = dict(bbox_inches="tight", dpi=150)

# 出力を保証する必須列（順序固定）
REQUIRED_OUT_COLS = [
    "step_length_r","step_length_l","step_asymmetry","speed_norm",
    "step_length_r_px","step_length_l_px","speed_px_s",
    "step_length_r_cm","step_length_l_cm","speed_m_s",
    "pelvis_sway","shoulder_diff_mean","shoulder_diff_std",
    "heel_tilt_r_mean","heel_tilt_l_mean","heel_tilt_r_std","heel_tilt_l_std",
    "knee_varus_valgus_r_mean","knee_varus_valgus_l_mean",
    "ankle_pronation_r_mean","ankle_pronation_l_mean"
]

# ===== ヘルパ =====
def _local_minima(y):
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1

def _enforce_min_distance(idxs, min_dist):
    if len(idxs) == 0:
        return idxs
    kept = [idxs[0]]
    for i in idxs[1:]:
        if i - kept[-1] >= min_dist:
            kept.append(i)
    return np.array(kept, dtype=int)

def _first_autocorr_peak(y, min_lag=8, max_lag=90):
    y = np.asarray(y) - np.nanmean(y)
    s = np.nanstd(y)
    if not np.isfinite(s) or s == 0 or len(y) < max_lag + 1:
        return None
    ac = np.correlate(y, y, mode='full')[len(y)-1:]
    ac = ac / (ac[0] if ac[0] != 0 else 1.0)
    seg = ac[min_lag:max_lag]
    if len(seg) == 0:
        return None
    return int(min_lag + np.argmax(seg))

def detect_steps_auto(y, min_distance=MIN_STEP_DISTANCE):
    y = np.asarray(y)
    mins = _local_minima(y)
    for p in PERCENTILES_TRY:
        thr = np.nanpercentile(y, p)
        cand = mins[y[mins] < thr]
        cand = _enforce_min_distance(np.sort(cand), min_distance)
        if len(cand) >= 2:
            return cand, thr
    return np.array([], dtype=int), None

def robust_speed_from_period(step_length_mean, period_frames, fps=FPS):
    if period_frames is None or period_frames <= 0:
        return 0.0
    cadence = fps / period_frames
    return float(step_length_mean * cadence)

def _first(series, default):
    try:
        v = float(series.iloc[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _maybe_to_px(arr, axis, img_w, img_h):
    """0..1 正規化を自動判定して px に換算。すでにpxならそのまま。"""
    if arr is None or len(arr) == 0:
        return arr
    mx = np.nanmax(arr)
    if np.isfinite(mx) and mx <= 1.5:  # 0..1(±α)
        scale = img_w if axis == 'x' else img_h
        if np.isfinite(scale) and scale > 0:
            return arr * scale
    return arr

def _ensure_all_required(results_dict):
    """必須列を欠損なく埋める（None/NaN→0.0, 欠損キー→0.0）"""
    for k in REQUIRED_OUT_COLS:
        v = results_dict.get(k, 0.0)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            v = 0.0
        results_dict[k] = v
    return results_dict

# ===== スムージング（Savitzky-Golay があれば使用、なければ移動平均） =====
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

def _smooth_1d(y: np.ndarray, win: int = 11, poly: int = 2) -> np.ndarray:
    """1次元信号をなめらかに。SciPyがあればsavgol、無ければ移動平均。"""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    # 窓長は奇数・長さ以下に調整
    win = max(3, min(win, len(y) - (1 - len(y) % 2)))
    if win % 2 == 0:
        win = max(3, win - 1)
    if _HAS_SAVGOL and len(y) >= win and win > poly:
        try:
            return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
        except Exception:
            pass
    # フォールバック：移動平均（中心）
    return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().to_numpy()

def _plot_raw_and_smooth(y, title, path, x=None, win=11, poly=2, xlabel="frame", ylabel="value"):
    """生データ(薄い線)＋平滑線(濃い線)の2本だけを描画（マーカー無し）。"""
    if y is None:
        y = []
    y = np.asarray(y, dtype=float)
    y_s = _smooth_1d(y, win=win, poly=poly) if len(y) > 2 else y

    fig = plt.figure(figsize=(10, 4))
    try:
        if x is None:
            plt.plot(y, linewidth=1.0, alpha=0.35, linestyle='-')
            plt.plot(y_s, linewidth=2.2, alpha=1.0, linestyle='-')
        else:
            plt.plot(x, y, linewidth=1.0, alpha=0.35, linestyle='-')
            plt.plot(x, y_s, linewidth=2.2, alpha=1.0, linestyle='-')
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, **PNGSAVE_KW)
    finally:
        plt.close(fig)

def _plot_double_raw_smooth(y1, y2, label1, label2, title, path, x=None, win=11, poly=2):
    """2系列（例：右踵Y/左踵Y）を、各々 生=薄線 + 平滑=濃線 で同じAxesに重ねる。"""
    y1 = np.asarray(y1 if y1 is not None else [], dtype=float)
    y2 = np.asarray(y2 if y2 is not None else [], dtype=float)
    s1 = _smooth_1d(y1, win=win, poly=poly) if len(y1) > 2 else y1
    s2 = _smooth_1d(y2, win=win, poly=poly) if len(y2) > 2 else y2
    fig = plt.figure(figsize=(10, 4))
    try:
        if x is None:
            plt.plot(y1, linewidth=1.0, alpha=0.30, linestyle='-', label=f"{label1} raw")
            plt.plot(s1, linewidth=2.0, alpha=1.0,  linestyle='-', label=f"{label1} smooth")
            plt.plot(y2, linewidth=1.0, alpha=0.30, linestyle='-', label=f"{label2} raw")
            plt.plot(s2, linewidth=2.0, alpha=1.0,  linestyle='-', label=f"{label2} smooth")
        else:
            plt.plot(x, y1, linewidth=1.0, alpha=0.30, linestyle='-', label=f"{label1} raw")
            plt.plot(x, s1, linewidth=2.0, alpha=1.0,  linestyle='-', label=f"{label1} smooth")
            plt.plot(x, y2, linewidth=1.0, alpha=0.30, linestyle='-', label=f"{label2} raw")
            plt.plot(x, s2, linewidth=2.0, alpha=1.0,  linestyle='-', label=f"{label2} smooth")
        plt.title(title); plt.xlabel("frame"); plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, **PNGSAVE_KW)
    finally:
        plt.close(fig)

def _plot_with_event_lines(y, strikes, title, path):
    """イベント縦線（ヒールストライク）を重ねた簡易版プロット"""
    y = np.asarray(y if y is not None else [], dtype=float)
    fig = plt.figure(figsize=(10, 4))
    try:
        plt.plot(y, linewidth=1.2)
        for s in strikes:
            plt.axvline(s, linestyle="--", linewidth=1)
        plt.title(title); plt.xlabel("frame"); plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(path, **PNGSAVE_KW)
    finally:
        plt.close(fig)

# ====== 本体 ======
def compute_metrics(angle_csv_path):
    df = pd.read_csv(angle_csv_path).reset_index(drop=True)
    video_dir = Path(angle_csv_path).parent
    plot_dir = video_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # ---- QC 初期化 ----
    qc = {
        "rows_raw": int(len(df)),
        "interp_after": None,
        "trim_head_ratio": 0.05,
        "outlier_sigma": 3.0,
        "scale_cm_per_px": None,
        "heel_strikes_right": [],
        "heel_strikes_left": [],
    }

    # ---- 入口 前処理（補間→冒頭5%トリム→再補間→3σクリップ）----
    df = interpolate_df(df)
    df = trim_head(df, ratio=0.05)
    df = interpolate_df(df)
    df = clip_df_by_sigma(df, sigma=3.0)
    qc["interp_after"] = int(len(df))

    results = {}

    # x/y/z系は補間＋平滑化（既存処理を温存）
    for col in df.columns:
        if col.endswith(('_x','_y','_z')):
            df[col] = df[col].interpolate(limit_direction='both')
            if SMOOTH_WINDOW > 1:
                df[col] = df[col].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

    # 画像サイズとスケール
    img_w = _first(df.get("img_w", pd.Series([np.nan])), np.nan)
    img_h = _first(df.get("img_h", pd.Series([np.nan])), np.nan)
    cm_per_px = _first(df.get("scale_cm_per_px", pd.Series([np.nan])), np.nan)
    # 未提供時：縦動画想定の暫定値（画面高 ≈ 100cm）
    if not np.isfinite(cm_per_px):
        if np.isfinite(img_h) and (not np.isfinite(img_w) or img_h >= img_w):
            cm_per_px = 100.0 / max(1.0, img_h)
        else:
            cm_per_px = 100.0 / 1080.0
    qc["scale_cm_per_px"] = float(cm_per_px)

    # ===== ① ステップ指標（ZがあればZ優先 / なければX/Y自動判定） =====
    dbg = {"axis": None, "thr_r": None, "thr_l": None,
           "period_frames_used": None, "img_w": img_w, "img_h": img_h,
           "scale_cm_per_px": cm_per_px, "z_available": False,
           "z_cols_present": False, "z_nan_ratio": None}
    try:
        # --- 入力（L/Rヒールの2D）
        y_rheel = df["28_y"].to_numpy()
        y_lheel = df["27_y"].to_numpy()
        x_rheel = df["28_x"].to_numpy()
        x_lheel = df["27_x"].to_numpy()
        # 0..1 → px
        x_rheel = _maybe_to_px(x_rheel, 'x', img_w, img_h)
        x_lheel = _maybe_to_px(x_lheel, 'x', img_w, img_h)
        y_rheel = _maybe_to_px(y_rheel, 'y', img_w, img_h)
        y_lheel = _maybe_to_px(y_lheel, 'y', img_w, img_h)

        # --- ここで“ヒールYの正規化”＆イベント検出（接地=0に平行移動）
        strikes_r = detect_heel_strikes(pd.Series(y_rheel), prominence=1.0)
        strikes_l = detect_heel_strikes(pd.Series(y_lheel), prominence=1.0)
        qc["heel_strikes_right"] = [int(x) for x in strikes_r]
        qc["heel_strikes_left"]  = [int(x) for x in strikes_l]
        df["right_heel_y_norm"] = normalize_heel_y_by_step(df.assign(tmp=y_rheel), "tmp", strikes_r).to_numpy()
        df["left_heel_y_norm"]  = normalize_heel_y_by_step(df.assign(tmp=y_lheel), "tmp", strikes_l).to_numpy()
        # ステップイベント重ね描画（参考）
        _plot_with_event_lines(y_rheel, strikes_r, "Right Heel Y with Events", plot_dir / "right_heel_y_events.png")
        _plot_with_event_lines(y_lheel, strikes_l, "Left Heel Y with Events",  plot_dir / "left_heel_y_events.png")

        # --- pelvis_z の自動生成（足Z or 股関節Z）
        if "pelvis_z" not in df.columns:
            try:
                if "23_z" in df.columns and "24_z" in df.columns:
                    df["pelvis_z"] = 0.5*(df["23_z"].astype(float) + df["24_z"].astype(float))
                elif "27_z" in df.columns and "28_z" in df.columns:
                    df["pelvis_z"] = 0.5*(df["27_z"].astype(float) + df["28_z"].astype(float))
                df["pelvis_z"] = df["pelvis_z"].interpolate(limit_direction="both")
            except Exception:
                pass

        # --- Z使用可否
        z_available = all(c in df.columns for c in ["pelvis_z","27_z","28_z"])
        dbg["z_cols_present"] = z_available
        if z_available:
            z_pelvis = df["pelvis_z"].astype(float).to_numpy()
            z_rheel  = df["28_z"].astype(float).to_numpy()
            z_lheel  = df["27_z"].astype(float).to_numpy()
            ok = (np.any(np.isfinite(z_pelvis)) and np.any(np.isfinite(z_rheel)) and np.any(np.isfinite(z_lheel)))
            if not ok:
                z_available = False
            else:
                dbg["z_nan_ratio"] = float(np.mean(~np.isfinite(z_pelvis)))
        dbg["z_available"] = bool(z_available)

        # --- 軸の決定とプロジェクション（cmで一貫）
        if z_available:
            axis = 'z'
            heel_r_proj_cm = z_rheel * 100.0   # m → cm想定
            heel_l_proj_cm = z_lheel * 100.0
            pelvis_proj_cm = z_pelvis * 100.0
        else:
            # 骨盤の純移動量で X/Y 判定（フォールバックつき）
            hip_lx = df.get("23_x", pd.Series(np.nan, index=df.index)).to_numpy()
            hip_rx = df.get("24_x", pd.Series(np.nan, index=df.index)).to_numpy()
            hip_ly = df.get("23_y", pd.Series(np.nan, index=df.index)).to_numpy()
            hip_ry = df.get("24_y", pd.Series(np.nan, index=df.index)).to_numpy()
            hip_lx = _maybe_to_px(hip_lx, 'x', img_w, img_h)
            hip_rx = _maybe_to_px(hip_rx, 'x', img_w, img_h)
            hip_ly = _maybe_to_px(hip_ly, 'y', img_w, img_h)
            hip_ry = _maybe_to_px(hip_ry, 'y', img_w, img_h)
            pelvis_x = 0.5*(hip_lx + hip_rx)
            pelvis_y = 0.5*(hip_ly + hip_ry)

            def _robust_net_disp(a):
                a = np.asarray(a, float)
                if len(a) < 5 or not np.any(np.isfinite(a)): return 0.0
                n = len(a); k = max(1, int(n*0.1))
                return float(abs(np.nanmedian(a[-k:]) - np.nanmedian(a[:k])))

            disp_x = _robust_net_disp(pelvis_x)
            disp_y = _robust_net_disp(pelvis_y)
            if disp_x > disp_y * 1.1:
                axis = 'x'
            elif disp_y > disp_x * 1.1:
                axis = 'y'
            else:
                rng_x = (np.nanmax([np.nanmax(x_rheel), np.nanmax(x_lheel)]) -
                         np.nanmin([np.nanmin(x_rheel), np.nanmin(x_lheel)]))
                rng_y = (np.nanmax([np.nanmax(y_rheel), np.nanmax(y_lheel)]) -
                         np.nanmin([np.nanmin(y_rheel), np.nanmin(y_lheel)]))
                axis = 'x' if (np.isfinite(rng_x) and rng_x >= rng_y) else 'y'

            heel_r_proj_cm = (x_rheel if axis=='x' else y_rheel) * cm_per_px
            heel_l_proj_cm = (x_lheel if axis=='x' else y_lheel) * cm_per_px
            pelvis_proj_cm = (pelvis_x if axis=='x' else pelvis_y) * cm_per_px

        dbg["axis"] = axis

        # --- 接地検出（既存の自動しきい値方式も併用）
        steps_r, thr_r = detect_steps_auto(y_rheel)
        steps_l, thr_l = detect_steps_auto(y_lheel)
        dbg["thr_r"] = float(thr_r) if thr_r is not None else None
        dbg["thr_l"] = float(thr_l) if thr_l is not None else None

        # 片側不足なら自己相関で補完
        pr = _first_autocorr_peak(y_rheel)
        pl = _first_autocorr_peak(y_lheel)

        def synth(y, period):
            if not period or period <= 0:
                return np.array([], dtype=int)
            mins = _local_minima(y)
            if len(mins) == 0:
                return np.array([], dtype=int)
            s0 = int(mins[0])
            return _enforce_min_distance(np.array(list(range(s0, len(y), int(period))), dtype=int),
                                         MIN_STEP_DISTANCE)
        if len(steps_r) < 2 and pr: steps_r = synth(y_rheel, pr)
        if len(steps_l) < 2 and pl: steps_l = synth(y_lheel, pl)

        def pick(x, idxs):
            if len(idxs) == 0: return np.array([])
            idxs = np.clip(idxs, 0, len(x)-1); return x[idxs]

        # --- ステップ長（cm）
        step_length_r_cm = step_length_l_cm = None
        if len(steps_r) >= 2:
            sr = pick(heel_r_proj_cm, steps_r)
            if len(sr) >= 2: step_length_r_cm = float(np.nanmedian(np.abs(np.diff(sr))))
        if len(steps_l) >= 2:
            sl = pick(heel_l_proj_cm, steps_l)
            if len(sl) >= 2: step_length_l_cm = float(np.nanmedian(np.abs(np.diff(sl))))

        # 代替
        if step_length_r_cm is None and step_length_l_cm is not None: step_length_r_cm = float(step_length_l_cm)
        if step_length_l_cm is None and step_length_r_cm is not None: step_length_l_cm = float(step_length_r_cm)
        if step_length_r_cm is None:
            rng = float(np.nanmax(heel_r_proj_cm) - np.nanmin(heel_r_proj_cm))
            step_length_r_cm = rng/3.0 if np.isfinite(rng) else 0.0
        if step_length_l_cm is None:
            rng = float(np.nanmax(heel_l_proj_cm) - np.nanmin(heel_l_proj_cm))
            step_length_l_cm = rng/3.0 if np.isfinite(rng) else 0.0

        denom = max((step_length_r_cm + step_length_l_cm)/2.0, 1e-6)
        step_asymmetry = float(abs(step_length_r_cm - step_length_l_cm) / denom)

        # --- 周期（同側差分の中央値／フォールバックに pr,pl）
        periods = []
        if len(steps_r) >= 2: periods.extend(np.diff(np.sort(steps_r)).tolist())
        if len(steps_l) >= 2: periods.extend(np.diff(np.sort(steps_l)).tolist())
        period_frames = float(np.median(periods)) if len(periods) else float(pr or pl or 0)
        dbg["period_frames_used"] = period_frames

        # --- 速度（cm/s基準 → m/s）
        N = len(pelvis_proj_cm)
        if N >= 2:
            speed_cm_s_traj = float(abs(pelvis_proj_cm[-1] - pelvis_proj_cm[0]) * FPS / (N-1))
        else:
            speed_cm_s_traj = 0.0
        step_len_mean_cm = float((step_length_r_cm + step_length_l_cm)/2.0)
        speed_cm_s_cad = float(robust_speed_from_period(step_len_mean_cm, period_frames, fps=FPS))
        speed_cm_s = float(np.median([v for v in [speed_cm_s_traj, speed_cm_s_cad] if np.isfinite(v)])) if (np.isfinite(speed_cm_s_traj) or np.isfinite(speed_cm_s_cad)) else 0.0
        speed_m_s = speed_cm_s / 100.0

        # --- px系も必ず埋める（Zモードでも cm→px で算出）
        if np.isfinite(cm_per_px) and cm_per_px > 0:
            step_length_r_px = step_length_r_cm / cm_per_px
            step_length_l_px = step_length_l_cm / cm_per_px
            speed_px_s = speed_cm_s / cm_per_px
        else:
            step_length_r_px = step_length_l_px = speed_px_s = 0.0

        # speed_norm は：Zモード→m/s、XYモード→px/s として統一出力
        speed_norm = round(speed_m_s if dbg["axis"] == 'z' else speed_px_s, 6)

        results.update({
            "step_length_r": round(step_length_r_cm, 6),   # cm
            "step_length_l": round(step_length_l_cm, 6),   # cm
            "step_asymmetry": round(step_asymmetry, 6),
            "speed_norm": speed_norm,
            "step_length_r_px": round(step_length_r_px, 3),
            "step_length_l_px": round(step_length_l_px, 3),
            "speed_px_s": round(speed_px_s, 3),
            "step_length_r_cm": round(step_length_r_cm, 2),
            "step_length_l_cm": round(step_length_l_cm, 2),
            "speed_m_s": round(speed_m_s, 3),
        })

        # --- steps.png（右・左の踵Y：各 raw+smooth を1枚に重ねて線のみ）
        _plot_double_raw_smooth(
            y_rheel, y_lheel, "Right Heel Y", "Left Heel Y",
            f"Heel Y Position (Axis={dbg['axis'].upper() if dbg['axis'] else 'NA'})",
            plot_dir / "steps.png"
        )

        # デバッグ出力
        pd.DataFrame({
            "axis":[dbg["axis"]],
            "z_available":[dbg["z_available"]],
            "z_cols_present":[dbg["z_cols_present"]],
            "z_nan_ratio":[dbg["z_nan_ratio"]],
            "steps_r":[list(map(int, steps_r))],
            "steps_l":[list(map(int, steps_l))],
            "thr_r":[dbg["thr_r"]], "thr_l":[dbg["thr_l"]],
            "period_frames_used":[dbg["period_frames_used"]],
            "img_w":[img_w], "img_h":[img_h], "scale_cm_per_px":[cm_per_px],
        }).to_csv(video_dir / "steps_debug.csv", index=False)

    except Exception as e:
        print(f"[!] Step metrics error: {e}")
        results.update({
            "step_length_r":0.0,"step_length_l":0.0,"step_asymmetry":0.0,
            "speed_norm":0.0,"step_length_r_px":0.0,"step_length_l_px":0.0,
            "speed_px_s":0.0,"step_length_r_cm":0.0,"step_length_l_cm":0.0,"speed_m_s":0.0
        })

    # ===== ② 骨盤スウェイ（中心xの標準偏差；px）→ 線のみ（従来値を維持）
    try:
        hip_center_x = (df.get("24_x", pd.Series(np.nan, index=df.index)) +
                        df.get("23_x", pd.Series(np.nan, index=df.index))) / 2
        hip_center_x = _maybe_to_px(hip_center_x.to_numpy(), 'x', img_w, img_h)
        results["pelvis_sway"] = round(float(np.nanstd(hip_center_x)), 6)
        _plot_raw_and_smooth(hip_center_x, "Pelvic Lateral Sway", plot_dir / "pelvis_sway.png")
    except Exception as e:
        print(f"[!] Pelvis sway error: {e}")
        results["pelvis_sway"] = 0.0
        _plot_raw_and_smooth([], "Pelvic Lateral Sway", plot_dir / "pelvis_sway.png")

    # --- 追加：cm換算＆中心化した骨盤スウェイ波形を別PNGとして保存（評価用）
    try:
        df_tmp = pd.DataFrame({
            "left_hip_x": df.get("23_x", pd.Series(np.nan, index=df.index)),
            "right_hip_x": df.get("24_x", pd.Series(np.nan, index=df.index)),
        })
        sway_cm_series = pelvic_lateral_sway_cm(df_tmp.rename(columns={
            "left_hip_x":"left_hip_x", "right_hip_x":"right_hip_x"
        }), cm_per_px)
        _plot_raw_and_smooth(sway_cm_series.to_numpy(), "Pelvic Lateral Sway (cm, centered)", plot_dir / "pelvic_lateral_sway_cm.png", ylabel="cm")
    except Exception as e:
        print(f"[!] Pelvis sway (cm) plot error: {e}")

    # ===== ③ 肩の高さ差（|Ly - Ry|；px）→ 線のみ
    try:
        shoulder_diff = np.abs(
            df.get("12_y", pd.Series(np.nan, index=df.index)) -
            df.get("11_y", pd.Series(np.nan, index=df.index))
        )
        shoulder_diff = _maybe_to_px(shoulder_diff.to_numpy(), 'y', img_w, img_h)
        results["shoulder_diff_mean"] = round(float(np.nanmean(shoulder_diff)), 6)
        results["shoulder_diff_std"]  = round(float(np.nanstd(shoulder_diff)), 6)
        _plot_raw_and_smooth(shoulder_diff, "Shoulder Height Asymmetry", plot_dir / "shoulder_diff.png")
    except Exception as e:
        print(f"[!] Shoulder diff error: {e}")
        results["shoulder_diff_mean"] = 0.0
        results["shoulder_diff_std"]  = 0.0
        _plot_raw_and_smooth([], "Shoulder Height Asymmetry", plot_dir / "shoulder_diff.png")

    # ===== ④ かかとZ（平均＋標準偏差）→ 右Zを線のみで
    try:
        z_rheel = df.get("28_z", pd.Series(np.nan, index=df.index)).to_numpy()
        z_lheel = df.get("27_z", pd.Series(np.nan, index=df.index)).to_numpy()
        results["heel_tilt_r_mean"] = round(float(np.nanmean(z_rheel)), 6)
        results["heel_tilt_l_mean"] = round(float(np.nanmean(z_lheel)), 6)
        results["heel_tilt_r_std"]  = round(float(np.nanstd(z_rheel)), 6)
        results["heel_tilt_l_std"]  = round(float(np.nanstd(z_lheel)), 6)
        _plot_raw_and_smooth(z_rheel, "Heel Tilt Z (Right)", plot_dir / "heel_tilt.png")
    except Exception as e:
        print(f"[!] Heel tilt error: {e}")
        results["heel_tilt_r_mean"] = 0.0; results["heel_tilt_l_mean"] = 0.0
        results["heel_tilt_r_std"]  = 0.0; results["heel_tilt_l_std"]  = 0.0
        _plot_raw_and_smooth([], "Heel Tilt Z (Right)", plot_dir / "heel_tilt.png")

    # ===== ⑤ 膝（XY）→ 線のみ
    try:
        if "knee_xy_r" in df.columns and "knee_xy_l" in df.columns:
            r = float(np.nanmean(df["knee_xy_r"]))
            l = float(np.nanmean(df["knee_xy_l"]))
            series_to_plot = df["knee_xy_r"].to_numpy()  # 右系列を代表で保存
        elif "knee_xy" in df.columns:
            r = l = float(np.nanmean(df["knee_xy"]))
            series_to_plot = df["knee_xy"].to_numpy()
        else:
            r = l = 0.0
            series_to_plot = np.array([])
        results["knee_varus_valgus_r_mean"] = round(r, 6)
        results["knee_varus_valgus_l_mean"] = round(l, 6)
        _plot_raw_and_smooth(series_to_plot, "Knee Varus/Valgus (XY)", plot_dir / "knee_varus_valgus.png")
    except Exception as e:
        print(f"[!] Knee angle error: {e}")
        _plot_raw_and_smooth([], "Knee Varus/Valgus (XY)", plot_dir / "knee_varus_valgus.png")

    # ===== ⑥ 足関節（XY）→ 線のみ
    try:
        if "ankle_xy_r" in df.columns and "ankle_xy_l" in df.columns:
            r = float(np.nanmean(df["ankle_xy_r"]))
            l = float(np.nanmean(df["ankle_xy_l"]))
            series_to_plot = df["ankle_xy_r"].to_numpy()  # 右系列を代表で保存
        elif "ankle_xy" in df.columns:
            r = l = float(np.nanmean(df["ankle_xy"]))
            series_to_plot = df["ankle_xy"].to_numpy()
        else:
            r = l = 0.0
            series_to_plot = np.array([])
        results["ankle_pronation_r_mean"] = round(r, 6)
        results["ankle_pronation_l_mean"] = round(l, 6)
        _plot_raw_and_smooth(series_to_plot, "Ankle Pronation/Supination (XY)", plot_dir / "ankle_pronation.png")
    except Exception as e:
        print(f"[!] Ankle angle error: {e}")
        _plot_raw_and_smooth([], "Ankle Pronation/Supination (XY)", plot_dir / "ankle_pronation.png")

    # ===== QC レポート保存 =====
    try:
        with open(video_dir / "qc_report.json", "w", encoding="utf-8") as f:
            json.dump(qc, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] qc_report.json save error: {e}")

    # ===== 必須列の欠損ゼロ化 =====
    results = _ensure_all_required(results)
    return results

# ====== CLI ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(" 使用法: python gait_metrics_calc.py /path/to/video_folder")
        sys.exit(1)
    target_folder = Path(sys.argv[1])
    angle_csv = target_folder / "angle_data.csv"
    if not angle_csv.exists():
        print(f" angle_data.csv が見つかりません: {angle_csv}")
        sys.exit(1)

    print(f"指標計算中: {angle_csv}")
    metrics = compute_metrics(angle_csv)
    # 必須列の順で保存（欠損ゼロ保証）
    pd.DataFrame([metrics], columns=REQUIRED_OUT_COLS).to_csv(target_folder / "gait_metrics.csv", index=False)
    print(f" gait_metrics.csv 保存完了 → {target_folder/'gait_metrics.csv'}")
    print(f" steps_debug.csv 保存完了 → {target_folder/'steps_debug.csv'}")
    print(f" qc_report.json 保存完了 → {target_folder/'qc_report.json'}")
