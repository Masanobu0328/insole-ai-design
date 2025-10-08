# -*- coding: utf-8 -*-
"""
front/side の angle_data.csv を時間合わせして統合します。
使い方:
  python gait_analysis/merge_two_views.py --front_dir <.../walk_front> \
                                          --side_dir  <.../walk_side>  \
                                          --out_dir   <.../walk_merged>

出力:
  <out_dir>/angle_data.csv          ← 統合版（以降 gait_metrics_calc.py がそのまま使える）
  <out_dir>/_merge_log.txt          ← 採用ルール、ラグ情報、採用率など
  <out_dir>/merge_qc.json           ← ラグ・列ごとの採用重み等のQC
  <out_dir>/sync_check.png          ← 同期確認プロット（front/sideのヒールY）
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------- 列名ヘルパ ---------
def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def smooth(x, w=7):
    if x is None:
        return None
    if len(x) == 0:
        return x
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    if np.all(np.isnan(x)):
        return x
    w = max(1, int(w))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

# --------- 同期（相互相関） ---------
def estimate_lag(a, b, max_lag=90):
    """a, b: 1D arrays (NaNは補間)。b が a に対して遅れている(+lag)/進んでいる(-lag)を推定"""
    if a is None or b is None:
        return 0
    A = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    B = pd.Series(b).interpolate(limit_direction="both").to_numpy()
    n = min(len(A), len(B))
    if n < 16:
        return 0
    A, B = A[:n], B[:n]
    A = (A - np.nanmean(A)) / (np.nanstd(A) + 1e-9)
    B = (B - np.nanmean(B)) / (np.nanstd(B) + 1e-9)
    best, bestlag = -1e9, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a2, b2 = A[lag:], B[:n - lag]
        else:
            a2, b2 = A[:n + lag], B[-lag:]
        if len(a2) < 16:
            continue
        c = np.nanmean(a2 * b2)
        if c > best:
            best, bestlag = c, lag
    return bestlag

def apply_lag(df, lag):
    """正: dfを遅らせる（先頭NaN）"""
    df = df.reset_index(drop=True)
    if lag == 0:
        return df
    if lag > 0:
        head = pd.DataFrame({c: [np.nan] * lag for c in df.columns})
        return pd.concat([head, df], ignore_index=True).iloc[:len(df)].reset_index(drop=True)
    # lag < 0: 進める（末尾NaN）
    out = df.iloc[-lag:].reset_index(drop=True)
    tail = pd.DataFrame({c: [np.nan] * (-lag) for c in df.columns})
    out = pd.concat([out, tail], ignore_index=True)
    return out.iloc[:len(df)].reset_index(drop=True)

# --------- マージの優先ルール ---------
FRONT_PREF = set([
    # 横成分・XY角度系（正面が有利）
    "pelvis_sway",
    "knee_varus_valgus_r","knee_varus_valgus_l","knee_varus_valgus",
    "ankle_pronation_r","ankle_pronation_l","ankle_pronation",
    "shoulder_diff","shoulder_diff_mean","shoulder_diff_std","shoulder_asym",
])
SIDE_PREF = set([
    # 上下・進行方向（側面が有利）
    "step_length_r_px","step_length_l_px","step_length_r_cm","step_length_l_cm",
    "step_length_r","step_length_l","speed_px_s","speed_m_s","speed_norm",
    "heel_tilt_r","heel_tilt_l","heel_tilt_z_r","heel_tilt_z_l",
    "toe_y_r","toe_y_l",
    # 高さ・前後（踵やつま先のY/Z）
    "27_y","28_y","27_z","28_z",
    "right_heel_y","left_heel_y","right_heel_y_norm","left_heel_y_norm",
])

# ヒールの上下動（同期用）の候補名（正規化を最優先）
HEEL_R_CANDS = ["right_heel_y_norm", "27_y", "right_heel_y", "heel_r_y", "r_heel_y", "RHEEL_y", "right_heel_y_px"]
HEEL_L_CANDS = ["left_heel_y_norm",  "28_y", "left_heel_y",  "heel_l_y", "l_heel_y", "LHEEL_y",  "left_heel_y_px"]

# --------- 重み付きマージ ---------
def weighted_merge(dfF, dfS, front_pref=FRONT_PREF, side_pref=SIDE_PREF):
    """
    同名列が front/side 両方にある場合：
      - front優先列: front 0.7, side 0.3
      - side優先列 : front 0.3, side 0.7
      - その他      : front 0.6, side 0.4
    片側のみ存在：そのまま採用
    欠損: 優先側→劣後側の順で埋め
    """
    n = min(len(dfF), len(dfS))
    F = dfF.iloc[:n].reset_index(drop=True)
    S = dfS.iloc[:n].reset_index(drop=True)
    merged = pd.DataFrame(index=range(n))
    stats = {"columns": {}}

    all_cols = list(dict.fromkeys(list(F.columns) + list(S.columns)))  # 重複除去で順序保持

    for col in all_cols:
        f_has = col in F.columns
        s_has = col in S.columns

        if not f_has and not s_has:
            continue

        if f_has and not s_has:
            merged[col] = F[col]
            stats["columns"][col] = {"source": "front_only"}
            continue

        if s_has and not f_has:
            merged[col] = S[col]
            stats["columns"][col] = {"source": "side_only"}
            continue

        # 両方にある場合は重み付き
        # ベース名でルール決定（_smooth/_mean/_std 等の接尾辞は無視）
        base = col
        for suf in ["_smooth", "_mean", "_std"]:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break

        if base in front_pref:
            wf, ws = 0.7, 0.3
            pref = "front"
        elif base in side_pref:
            wf, ws = 0.3, 0.7
            pref = "side"
        else:
            wf, ws = 0.6, 0.4
            pref = "front*"

        f = F[col].to_numpy(dtype=float)
        s = S[col].to_numpy(dtype=float)

        # 数値配列（NaN可）として重み付き合成
        num = wf * f + ws * s
        den = wf + ws
        out = np.where(np.isfinite(num), num / den, np.nan)

        # 欠損が残る箇所は優先側で埋め、なお欠損なら劣後側で埋める
        out = pd.Series(out)
        if pref.startswith("front"):
            out = out.fillna(pd.Series(f)).fillna(pd.Series(s))
        else:
            out = out.fillna(pd.Series(s)).fillna(pd.Series(f))

        merged[col] = out.to_numpy()
        stats["columns"][col] = {
            "source": "weighted",
            "preferred": pref,
            "weights": {"front": wf, "side": ws},
            "front_avail_ratio": float(np.isfinite(f).mean()),
            "side_avail_ratio": float(np.isfinite(s).mean()),
        }

    return merged, stats

def plot_sync_check(sigF, sigS, lag, out_path):
    """同期確認図（front基準、sideはラグ適用後）"""
    if sigF is None or sigS is None:
        return
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(sigF, linewidth=1.5, label="front")
        plt.plot(sigS, linewidth=1.5, label=f"side (shifted {lag})")
        plt.title("Heel Y signals (sync check)")
        plt.xlabel("frame"); plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front_dir", type=str, required=True)
    ap.add_argument("--side_dir",  type=str, required=True)
    ap.add_argument("--out_dir",   type=str, required=True)
    args = ap.parse_args()

    front_dir = Path(args.front_dir)
    side_dir  = Path(args.side_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f_csv = front_dir / "angle_data.csv"
    s_csv = side_dir  / "angle_data.csv"
    if not f_csv.exists() or not s_csv.exists():
        raise SystemExit(f"[!] angle_data.csv が見つかりません\n  front:{f_csv}\n  side :{s_csv}")

    dfF = pd.read_csv(f_csv)
    dfS = pd.read_csv(s_csv)

    # ---- 同期信号の選択（右ヒール優先、なければ左） ----
    rF = pick_col(dfF, HEEL_R_CANDS); rS = pick_col(dfS, HEEL_R_CANDS)
    lF = pick_col(dfF, HEEL_L_CANDS); lS = pick_col(dfS, HEEL_L_CANDS)

    sigF = smooth(dfF[rF].to_numpy()) if rF else (smooth(dfF[lF].to_numpy()) if lF else None)
    sigS = smooth(dfS[rS].to_numpy()) if rS else (smooth(dfS[lS].to_numpy()) if lS else None)

    # ---- ラグ推定（frontを基準に、sideをシフト）----
    lag = estimate_lag(sigF, sigS, max_lag=90)
    dfF2 = apply_lag(dfF, 0)       # frontは基準
    dfS2 = apply_lag(dfS, lag)     # side を front に合わせる

    # 長さを揃える
    n = min(len(dfF2), len(dfS2))
    dfF2, dfS2 = dfF2.iloc[:n].reset_index(drop=True), dfS2.iloc[:n].reset_index(drop=True)

    # 同期確認図
    try:
        plot_sync_check(sigF[:n] if sigF is not None else None,
                        (smooth(dfS[rS].to_numpy()) if rS else (smooth(dfS[lS].to_numpy()) if lS else None))[:n] if (rS or lS) else None,
                        lag, out_dir / "sync_check.png")
    except Exception:
        pass

    # ---- 重み付きマージ ----
    merged, stats = weighted_merge(dfF2, dfS2, FRONT_PREF, SIDE_PREF)

    # 共通メタ（存在する方を採用）
    for meta in ["frame", "img_w", "img_h", "scale_cm_per_px", "axis"]:
        if meta not in merged.columns:
            if meta in dfF2.columns:
                merged[meta] = dfF2[meta]
            elif meta in dfS2.columns:
                merged[meta] = dfS2[meta]

    # 列順はfront→sideの登場順をなるべく維持
    merged = merged[[c for c in dict.fromkeys(list(dfF2.columns) + list(dfS2.columns) + list(merged.columns)) if c in merged.columns]]

    # ---- 出力 ----
    out_csv = out_dir / "angle_data.csv"
    merged.to_csv(out_csv, index=False)

    # ログ（人間向け）
    with open(out_dir / "_merge_log.txt", "w", encoding="utf-8") as f:
        f.write(f"front_dir: {front_dir}\nside_dir : {side_dir}\nout_dir  : {out_dir}\n")
        f.write(f"estimated_lag_frames (side relative to front): {lag}\n")
        f.write(f"final_length: {len(merged)} frames\n\n")
        f.write("== columns summary ==\n")
        for k, v in stats["columns"].items():
            if v.get("source") == "weighted":
                f.write(f"{k:32s} <- weighted ({v['preferred']})  wF={v['weights']['front']:.1f} wS={v['weights']['side']:.1f}  "
                        f"avail(F={v['front_avail_ratio']:.2f}, S={v['side_avail_ratio']:.2f})\n")
            else:
                f.write(f"{k:32s} <- {v['source']}\n")

    # QC（機械可読）
    qc = {
        "lag_applied_to_side": int(lag),
        "frames": int(len(merged)),
        "columns": stats["columns"],
        "signals_used": {
            "front_signal": rF or lF,
            "side_signal":  rS or lS
        }
    }
    with open(out_dir / "merge_qc.json", "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    print(f"[OK] merged -> {out_csv}  (lag={lag} frames, {len(merged)} rows)")
    print(f"     log   -> {out_dir/'_merge_log.txt'}")
    print(f"     qc    -> {out_dir/'merge_qc.json'}")
    print(f"     plot  -> {out_dir/'sync_check.png'}")

if __name__ == "__main__":
    main()
