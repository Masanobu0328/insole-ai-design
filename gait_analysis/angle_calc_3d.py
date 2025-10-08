# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# 追加：スケール推定を共通化
try:
    from .preprocess import load_scale_or_estimate
except Exception:
    # preprocess が未導入でも動くフォールバック
    def load_scale_or_estimate(df, scale_json_path=None):
        # 肩(12/11)→足首(28/27)の距離などから 170cm 仮定の簡易推定
        def med_dist(ax, ay, bx, by):
            if all(c in df.columns for c in [ax, ay, bx, by]):
                d = np.sqrt((df[ax] - df[bx]) ** 2 + (df[ay] - df[by]) ** 2)
                d = d.replace([np.inf, -np.inf], np.nan).dropna()
                if len(d) > 0:
                    return float(d.median())
            return None
        cands = []
        d1 = med_dist("12_x", "12_y", "28_x", "28_y")  # 右肩-右足首
        d2 = med_dist("11_x", "11_y", "27_x", "27_y")  # 左肩-左足首
        for d in [d1, d2]:
            if d and d > 0:
                cands.append(170.0 / d)  # 170cm 仮定
        return float(np.median(cands)) if cands else 0.2  # フォールバック=0.2 cm/px（5px=1cm想定）

# 縦動画のデフォルト（必要に応じて変更可）
DEFAULT_IMG_W = 1080.0
DEFAULT_IMG_H = 1920.0

# ---------------- 基本ユーティリティ ----------------
def calc_3d_angle(a, b, c):
    ba = a - b
    bc = c - b
    ba_norm = ba / (np.linalg.norm(ba) + 1e-6)
    bc_norm = bc / (np.linalg.norm(bc) + 1e-6)
    cos_angle = float(np.dot(ba_norm, bc_norm))
    angle_3d = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    def angle_on_plane(v1, v2, axis='xy'):
        if axis == 'xy':
            v1p, v2p = v1[:2], v2[:2]
        elif axis == 'yz':
            v1p, v2p = v1[1:], v2[1:]
        elif axis == 'xz':
            v1p, v2p = np.array([v1[0], v1[2]]), np.array([v2[0], v2[2]])
        else:
            raise ValueError("Invalid axis")
        denom = (np.linalg.norm(v1p) * np.linalg.norm(v2p) + 1e-6)
        cos_plane = float(np.dot(v1p, v2p)) / denom
        return float(np.degrees(np.arccos(np.clip(cos_plane, -1.0, 1.0))))

    return {
        '3d': angle_3d,
        'xy': angle_on_plane(ba_norm, bc_norm, 'xy'),
        'yz': angle_on_plane(ba_norm, bc_norm, 'yz'),
        'xz': angle_on_plane(ba_norm, bc_norm, 'xz')
    }

def safe_triplet(df, i, idx):
    x = df.get(f"{idx}_x", pd.Series([np.nan]*len(df))).iloc[i]
    y = df.get(f"{idx}_y", pd.Series([np.nan]*len(df))).iloc[i]
    z = df.get(f"{idx}_z", pd.Series([np.nan]*len(df))).iloc[i]
    x = np.nan if pd.isna(x) else float(x)
    y = np.nan if pd.isna(y) else float(y)
    z = np.nan if pd.isna(z) else float(z)
    return np.array([x, y, z], dtype=float)

def _get_first(series, fallback):
    try:
        v = float(series.iloc[0])
        return v if np.isfinite(v) else fallback
    except Exception:
        return fallback

# ---------------- 本体 ----------------
def compute_3d_joint_angles(csv_path, output_path, img_w=None, img_h=None, scale_cm_per_px=None):
    df = pd.read_csv(csv_path)

    # 画像サイズはCSV内が優先、無ければ引数→デフォルトの順
    if img_w is None:
        img_w = _get_first(df.get("img_w", pd.Series([np.nan])), DEFAULT_IMG_W)
    if img_h is None:
        img_h = _get_first(df.get("img_h", pd.Series([np.nan])), DEFAULT_IMG_H)

    # スケールは CSV内 or 引数 が無ければ推定して**必ず**確定
    if scale_cm_per_px is None:
        csv_scale = _get_first(df.get("scale_cm_per_px", pd.Series([np.nan])), np.nan)
        scale_cm_per_px = (csv_scale if np.isfinite(csv_scale) and csv_scale > 0
                           else load_scale_or_estimate(df, scale_json_path=None))
    # 安全下限（極端な値の暴走を防ぐ）
    if not np.isfinite(scale_cm_per_px) or scale_cm_per_px <= 0:
        scale_cm_per_px = 0.2  # 5px ≒ 1cm のフォールバック

    results = []

    # pelvis_z を出力に持たせる（後段の利用を安定化）
    # 1) 左右股関節Z(23_z,24_z) があれば中点、2) それが無ければ左右踵Z(27_z,28_z) の中点
    pelvis_z_series = None
    if all(c in df.columns for c in ["23_z", "24_z"]):
        pelvis_z_series = (df["23_z"].astype(float) + df["24_z"].astype(float)) / 2.0
    elif all(c in df.columns for c in ["27_z", "28_z"]):
        pelvis_z_series = (df["27_z"].astype(float) + df["28_z"].astype(float)) / 2.0
    if pelvis_z_series is not None:
        pelvis_z_series = pelvis_z_series.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")

    for i in range(len(df)):
        row = {'frame': i}
        try:
            for side, label in [
                ('R', {'shoulder':12, 'hip':24, 'knee':26, 'ankle':28, 'foot':32}),
                ('L', {'shoulder':11, 'hip':23, 'knee':25, 'ankle':27, 'foot':31})
            ]:
                sh = safe_triplet(df, i, label['shoulder'])
                hp = safe_triplet(df, i, label['hip'])
                kn = safe_triplet(df, i, label['knee'])
                an = safe_triplet(df, i, label['ankle'])
                ft = safe_triplet(df, i, label['foot'])

                hip_angles   = calc_3d_angle(sh, hp, kn)
                knee_angles  = calc_3d_angle(hp, kn, an)
                ankle_angles = calc_3d_angle(kn, an, ft)

                for k, v in hip_angles.items():
                    row[f"hip_{side}_{k}"] = v
                for k, v in knee_angles.items():
                    row[f"knee_{side}_{k}"] = v
                for k, v in ankle_angles.items():
                    row[f"ankle_{side}_{k}"] = v

            # 骨盤傾斜（左右股関節差）
            hip_r = safe_triplet(df, i, 24)
            hip_l = safe_triplet(df, i, 23)
            pelvis_vec = hip_r - hip_l
            denom = pelvis_vec[2] if abs(pelvis_vec[2]) > 1e-9 else 1e-9
            row["pelvic_yz"] = float(np.degrees(np.arctan2(pelvis_vec[1], denom)))
            row["pelvic_xz"] = float(np.degrees(np.arctan2(pelvis_vec[0], denom)))

            # gait_metrics_calc.py が使う座標（踵は x/y/z 必須）
            for k in ["28_x","28_y","28_z","27_x","27_y","27_z","24_x","23_x","11_y","12_y"]:
                row[k] = df.get(k, pd.Series([np.nan]*len(df))).iloc[i]

            # pelvis_z を書き出し（あれば）
            row["pelvis_z"] = float(pelvis_z_series.iloc[i]) if pelvis_z_series is not None else np.nan

            # 角度の左右別と平均
            row["knee_xy_r"]  = row.get("knee_R_xy", np.nan)
            row["knee_xy_l"]  = row.get("knee_L_xy", np.nan)
            row["ankle_xy_r"] = row.get("ankle_R_xy", np.nan)
            row["ankle_xy_l"] = row.get("ankle_L_xy", np.nan)
            row["knee_xy"]  = float(np.nanmean([row["knee_xy_r"],  row["knee_xy_l"]]))
            row["ankle_xy"] = float(np.nanmean([row["ankle_xy_r"], row["ankle_xy_l"]]))

            # 画像サイズとスケールを毎フレームへ
            row["img_w"] = float(img_w)
            row["img_h"] = float(img_h)
            row["scale_cm_per_px"] = float(scale_cm_per_px)

        except Exception:
            # 例外時も列欠落を防ぐ
            for side in ['R', 'L']:
                for joint in ['hip', 'knee', 'ankle']:
                    for axis in ['3d', 'xy', 'yz', 'xz']:
                        row[f"{joint}_{side}_{axis}"] = np.nan
            row["pelvic_yz"] = row["pelvic_xz"] = np.nan
            for k in ["28_x","28_y","28_z","27_x","27_y","27_z","24_x","23_x","11_y","12_y",
                      "knee_xy_r","knee_xy_l","ankle_xy_r","ankle_xy_l","knee_xy","ankle_xy",
                      "pelvis_z","img_w","img_h","scale_cm_per_px"]:
                row[k] = np.nan

        results.append(row)

    df_out = pd.DataFrame(results)

    # 列の存在を最終確認（欠けていたら追加）
    must_cols = ["28_x","28_y","28_z","27_x","27_y","27_z","24_x","23_x","11_y","12_y",
                 "knee_xy_r","knee_xy_l","ankle_xy_r","ankle_xy_l","knee_xy","ankle_xy",
                 "pelvis_z","img_w","img_h","scale_cm_per_px"]
    for c in must_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    # 数値化＆無限大をNaNへ
    for c in df_out.columns:
        if df_out[c].dtype == object:
            try:
                df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
            except Exception:
                pass
        df_out[c] = df_out[c].replace([np.inf, -np.inf], np.nan)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    return df_out

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="raw pose CSV")
    ap.add_argument("output_csv", help="angle_data.csv path")
    ap.add_argument("--img_w", type=float, default=None, help="frame width px (optional)")
    ap.add_argument("--img_h", type=float, default=None, help="frame height px (optional)")
    ap.add_argument("--scale_cm_per_px", type=float, default=None, help="cm per px (optional)")
    args = ap.parse_args()

    compute_3d_joint_angles(
        args.input_csv,
        args.output_csv,
        img_w=args.img_w,
        img_h=args.img_h,
        scale_cm_per_px=args.scale_cm_per_px
    )
