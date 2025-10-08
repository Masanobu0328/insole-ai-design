# gait_metrics_calc.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

def compute_metrics(angle_csv_path):
    df = pd.read_csv(angle_csv_path)
    results = {}
    video_dir = Path(angle_csv_path).parent
    plot_dir = video_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # === ① ステップ検出（ピーク検出）===
    def detect_steps(y_heel, threshold_ratio=0.95):
        y_heel = np.array(y_heel)
        norm = (y_heel - np.min(y_heel)) / (np.max(y_heel) - np.min(y_heel) + 1e-6)
        threshold = threshold_ratio * np.min(norm)
        steps = []
        for i in range(1, len(norm) - 1):
            if norm[i-1] > norm[i] < norm[i+1] and norm[i] < threshold:
                steps.append(i)
        return steps

    try:
        y_rheel = df["28_y"].values
        y_lheel = df["27_y"].values
        x_rheel = df["28_x"].values
        x_lheel = df["27_x"].values

        steps_r = detect_steps(y_rheel)
        steps_l = detect_steps(y_lheel)

        if len(steps_r) < 2 or len(steps_l) < 2:
            raise ValueError("ステップ数が不足しています")

        # 最も近いステップ同士でステップ長を計算
        min_len = min(len(steps_r), len(steps_l))
        step_pairs = list(zip(steps_r[:min_len], steps_l[:min_len]))
        step_lengths = [abs(x_rheel[r] - x_lheel[l]) for r, l in step_pairs]
        frame_interval = np.mean([abs(r - l) for r, l in step_pairs])

        fps = 30
        step_length = np.mean(step_lengths)
        speed = (step_length * fps) / frame_interval if frame_interval else None
        step_asymmetry = abs(len(steps_r) - len(steps_l)) / max(len(steps_r), 1)

        results["step_length"] = step_length
        results["step_asymmetry"] = step_asymmetry
        results["speed"] = speed

        plt.figure()
        plt.plot(y_rheel, label="Right Heel Y")
        plt.plot(y_lheel, label="Left Heel Y")
        plt.scatter(steps_r, y_rheel[steps_r], c='red', label='R steps')
        plt.scatter(steps_l, y_lheel[steps_l], c='blue', label='L steps')
        plt.title("Heel Step Detection")
        plt.legend()
        plt.savefig(plot_dir / "steps.png")
        plt.close()
    except Exception as e:
        print(f"[!] Step metrics error: {e}")
        results["step_length"] = None
        results["step_asymmetry"] = None
        results["speed"] = None

    # === 以下略（②〜⑥はそのまま）===
    # ※以下は元のコードのまま継続して使えます。変更不要。

    try:
        hip_center_x = (df["24_x"] + df["23_x"]) / 2
        results["pelvis_sway"] = np.std(hip_center_x)
        plt.figure()
        plt.plot(hip_center_x, label="Pelvis Center X")
        plt.title("Pelvic Lateral Sway")
        plt.legend()
        plt.savefig(plot_dir / "pelvis_sway.png")
        plt.close()
    except Exception as e:
        print(f"[!] Pelvis sway error: {e}")

    try:
        shoulder_diff = np.abs(df["12_y"] - df["11_y"])
        results["shoulder_diff_mean"] = np.mean(shoulder_diff)
        results["shoulder_diff_std"] = np.std(shoulder_diff)
        plt.figure()
        plt.plot(shoulder_diff, label="Shoulder Y Difference")
        plt.title("Shoulder Height Asymmetry")
        plt.legend()
        plt.savefig(plot_dir / "shoulder_diff.png")
        plt.close()
    except Exception as e:
        print(f"[!] Shoulder diff error: {e}")

    try:
        z_rheel = df["28_z"]
        results["heel_tilt_r_mean"] = np.mean(z_rheel)
        plt.figure()
        plt.plot(z_rheel, label="Right Heel Z")
        plt.title("Heel Tilt (Z)")
        plt.legend()
        plt.savefig(plot_dir / "heel_tilt.png")
        plt.close()
    except Exception as e:
        print(f"[!] Heel tilt error: {e}")

    try:
        results["knee_varus_valgus_mean"] = np.mean(df["knee_xy"])
        plt.figure()
        plt.plot(df["knee_xy"], label="Knee XY Angle")
        plt.title("Knee Varus/Valgus (XY Plane)")
        plt.legend()
        plt.savefig(plot_dir / "knee_varus_valgus.png")
        plt.close()
    except Exception as e:
        print(f"[!] Knee angle error: {e}")

    try:
        results["ankle_pronation_supination_mean"] = np.mean(df["ankle_xy"])
        plt.figure()
        plt.plot(df["ankle_xy"], label="Ankle XY Angle")
        plt.title("Ankle Pronation/Supination (XY Plane)")
        plt.legend()
        plt.savefig(plot_dir / "ankle_pronation.png")
        plt.close()
    except Exception as e:
        print(f"[!] Ankle angle error: {e}")

    return results


# ========================
# 実行部分
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(" 使用法: python gait_metrics_calc.py videos/output/your_video")
        sys.exit(1)

    target_folder = Path(sys.argv[1])
    angle_csv = target_folder / "angle_data.csv"
    if not angle_csv.exists():
        print(f" angle_data.csv が見つかりません: {angle_csv}")
        sys.exit(1)

    print(f"指標計算中: {angle_csv}")
    metrics = compute_metrics(angle_csv)
    df = pd.DataFrame([metrics])
    df.to_csv(target_folder / "gait_metrics.csv", index=False)
    print(f" gait_metrics.csv 保存完了 → {target_folder/'gait_metrics.csv'}")
