# lift3d.py
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch


# ---- 列マッピング候補（あなたの CSV をできるだけ吸収）
CAND = {
    # COCO index -> list of column name patterns for x,y
    # 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
    # 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
    # 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
    # 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
    0:  (["0_x","nose_x","Nose_x"], ["0_y","nose_y","Nose_y"]),
    1:  (["1_x","left_eye_x","leye_x"], ["1_y","left_eye_y","leye_y"]),
    2:  (["2_x","right_eye_x","reye_x"], ["2_y","right_eye_y","reye_y"]),
    3:  (["3_x","left_ear_x"], ["3_y","left_ear_y"]),
    4:  (["4_x","right_ear_x"], ["4_y","right_ear_y"]),
    5:  (["5_x","11_x","left_shoulder_x"], ["5_y","11_y","left_shoulder_y"]),
    6:  (["6_x","12_x","right_shoulder_x"], ["6_y","12_y","right_shoulder_y"]),
    7:  (["7_x","left_elbow_x"], ["7_y","left_elbow_y"]),
    8:  (["8_x","right_elbow_x"], ["8_y","right_elbow_y"]),
    9:  (["9_x","left_wrist_x"], ["9_y","left_wrist_y"]),
    10: (["10_x","right_wrist_x"], ["10_y","right_wrist_y"]),
    11: (["11_x","23_x","left_hip_x"], ["11_y","23_y","left_hip_y"]),
    12: (["12_x","24_x","right_hip_x"], ["12_y","24_y","right_hip_y"]),
    13: (["13_x","left_knee_x"], ["13_y","left_knee_y"]),
    14: (["14_x","right_knee_x"], ["14_y","right_knee_y"]),
    15: (["15_x","27_x","left_ankle_x","left_heel_x"], ["15_y","27_y","left_ankle_y","left_heel_y"]),
    16: (["16_x","28_x","right_ankle_x","right_heel_x"], ["16_y","28_y","right_ankle_y","right_heel_y"]),
}

def pick(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def build_kp2d(df):
    """return [T,17,2] in pixels (自動で 0..1 → px に換算) と img_w, img_h"""
    # img size
    img_w = float(df.get("img_w", pd.Series([np.nan])).iloc[0])
    img_h = float(df.get("img_h", pd.Series([np.nan])).iloc[0])
    if not np.isfinite(img_w) or img_w <= 1: img_w = 1080.0
    if not np.isfinite(img_h) or img_h <= 1: img_h = 1920.0

    T = len(df)
    kp = np.full((T,17,2), np.nan, dtype=np.float32)

    for j in range(17):
        cx = pick(df, CAND[j][0])
        cy = pick(df, CAND[j][1])
        if cx is None or cy is None: continue
        x = df[cx].astype(float).to_numpy()
        y = df[cy].astype(float).to_numpy()
        # 0..1 → px
        if np.nanmax(x) <= 1.5: x = x * img_w
        if np.nanmax(y) <= 1.5: y = y * img_h
        # 線形補間
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        kp[:,j,0] = x
        kp[:,j,1] = y

    return kp, img_w, img_h

def load_videopose3d(vp3d_dir, ckpt_path, device):
    sys.path.insert(0, str(Path(vp3d_dir)/"src"))
    from common.model import TemporalModel
    # 17関節 / 受容野243 の構成（チェックポイントに合わせる）
    model = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024)
    ckpt = torch.load(ckpt_path, map_location=device)
    # 一般に 'model_pos' キーで保存されている
    state = ckpt['model_pos'] if 'model_pos' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model

@torch.no_grad()
def infer_3d(kp2d_px, img_w, img_h, model, device="cpu"):
    """kp2d_px: [T,17,2] pixels → return [T,17,3] arbitrary units"""
    kp = kp2d_px.copy().astype(np.float32)
    # 骨盤中心で平行移動
    pelvis = 0.5*(kp[:,11,:] + kp[:,12,:])  # L/R hip
    kp -= pelvis[:,None,:]
    # 画面サイズでスケール正規化（-2..2 程度）
    kp[...,0] /= max(1.0, img_w/2.0)
    kp[...,1] /= max(1.0, img_h/2.0)
    # 受容野pad
    RF = 243
    pad = (RF-1)//2
    kp_pad = np.pad(kp, ((pad,pad),(0,0),(0,0)), mode="edge")
    x = torch.from_numpy(kp_pad[None]).to(device)  # [1,T+2p,17,2]
    y = model(x)                                   # [1,T,17,3]
    j3d = y[0].cpu().numpy()
    return j3d  # 単位は相対（スケールなし）

def scale_to_metric(j3d, height_cm=170.0):
    """身長でスケール付与 → m単位で返す"""
    j = j3d.copy()
    T = j.shape[0]
    t0 = T//2
    # 身長近似: nose(0) と 両足首(15,16) の中点
    head = j[t0,0]
    ank  = 0.5*(j[t0,15] + j[t0,16])
    approx_h = np.linalg.norm(head - ank)
    scale = (height_cm/100.0) / max(approx_h, 1e-6)
    return j * scale

def append_z_columns(df, j3d_m):
    """df に 23_z,24_z,27_z,28_z, pelvis_z を追加（なければ作成）"""
    T = len(df)
    out = df.copy()
    # COCO: 11=L_hip, 12=R_hip, 15=L_ankle, 16=R_ankle
    LHIP_Z = j3d_m[:,11,2]
    RHIP_Z = j3d_m[:,12,2]
    LANK_Z = j3d_m[:,15,2]
    RANK_Z = j3d_m[:,16,2]
    PELV_Z = 0.5*(LHIP_Z + RHIP_Z)

    out["23_z"] = LHIP_Z
    out["24_z"] = RHIP_Z
    out["27_z"] = LANK_Z   # あなたの定義で 27=左踵として運用
    out["28_z"] = RANK_Z   # 28=右踵
    out["pelvis_z"] = PELV_Z
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_folder", type=str, help="angle_data.csv が入ったフォルダ")
    ap.add_argument("--vp3d_dir", type=str, required=True, help="VideoPose3D のフォルダパス")
    ap.add_argument("--ckpt", type=str, required=True, help="事前学習済みckptファイル（.bin等）")
    ap.add_argument("--height_cm", type=float, default=170.0, help="被験者の身長(cm)")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    folder = Path(args.video_folder)
    csv_path = folder/"angle_data.csv"
    if not csv_path.exists():
        print(f"[!] angle_data.csv が見つかりません: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    kp2d_px, img_w, img_h = build_kp2d(df)
    model = load_videopose3d(args.vp3d_dir, args.ckpt, args.device)
    j3d_rel = infer_3d(kp2d_px, img_w, img_h, model, device=args.device)
    j3d_m   = scale_to_metric(j3d_rel, height_cm=args.height_cm)

    df2 = append_z_columns(df, j3d_m)
    df2.to_csv(csv_path, index=False)  # 上書き保存（z列が増えます）
    print("3D化＆z列追加 完了:", csv_path)

if __name__ == "__main__":
    main()
