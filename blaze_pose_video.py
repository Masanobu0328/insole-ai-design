# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import cv2
import mediapipe as mp
import pandas as pd

# あなたの角度計算関数（ファイルレイアウトに合わせて import）
from gait_analysis.angle_calc_3d import compute_3d_joint_angles

def process_video(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = out_dir / "pose_output.mp4"
    pose_csv_path     = out_dir / "pose_data.csv"
    angle_csv_path    = out_dir / "angle_data.csv"

    # すでに angle_data.csv があるならスキップ（再計算したい時だけ削除 or --force運用）
    if angle_csv_path.exists():
        print(f"[skip] angle_data.csv 既に存在: {angle_csv_path}")
        return

    # MediaPipe Pose 初期化
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[!] 動画を開けませんでした: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1080
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_v  = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # 33ランドマーク * (x,y,z,visibility)
    landmark_cols = [f"{i}_{axis}" for i in range(33) for axis in ("x","y","z","vis")]
    # 追加のメタ列（後工程でスケール判定に使う）
    meta_cols = ["img_w", "img_h"]
    all_cols = landmark_cols + meta_cols

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # 可視化（出力動画）
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out_v.write(cv2.resize(frame, (width, height)))

        # CSV行の構築
        row = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            row = [None] * (33 * 4)

        # メタ列
        row.extend([float(width), float(height)])
        rows.append(row)
        frame_idx += 1

    cap.release()
    out_v.release()
    pose.close()

    # CSV保存
    df = pd.DataFrame(rows, columns=all_cols)
    df.to_csv(pose_csv_path, index=False)

    # ---- 3D関節角度などの派生CSVを生成（angle_calc_3d側の関数を利用）----
    angle_df = compute_3d_joint_angles(str(pose_csv_path), str(angle_csv_path))
    # もし compute_3d_joint_angles が返り値を返すだけなら保存する
    if isinstance(angle_df, pd.DataFrame):
        angle_df.to_csv(angle_csv_path, index=False)

    print(
        f" 処理完了: {video_path}\n"
        f" → {output_video_path}\n"
        f" → {pose_csv_path}\n"
        f" → {angle_csv_path}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path", type=str, help="入力動画のパス")
    ap.add_argument("out_dir", type=str, help="出力フォルダ（例: videos/output/<stem>）")
    args = ap.parse_args()

    video_path = Path(args.video_path).resolve()
    out_dir    = Path(args.out_dir).resolve()

    if not video_path.exists():
        print(f"[!] 入力動画が見つかりません: {video_path}")
        sys.exit(1)

    process_video(video_path, out_dir)

if __name__ == "__main__":
    main()
