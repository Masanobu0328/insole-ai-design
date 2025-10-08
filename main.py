# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path
import subprocess, shlex

def run(cmd, cwd=None, extra_env=None):
    print(">>", cmd, flush=True)
    if cmd.strip().startswith("python "):
        py = sys.executable or "python"
        cmd = cmd.replace("python ", f'"{py}" -u ', 1)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    ret = subprocess.call(shlex.split(cmd), cwd=cwd, env=env)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd} (exit={ret})")

def main():
    if len(sys.argv) < 2:
        print("使い方: python main.py <path/to/video.mp4> [out_dir]")
        sys.exit(1)

    video = Path(sys.argv[1]).resolve()
    if not video.exists():
        print(f"[!] 動画が見つかりません: {video}")
        sys.exit(1)

    # ★ 追加: 第2引数に出力先を受け取れるように
    if len(sys.argv) >= 3:
        out_dir = Path(sys.argv[2]).resolve()
    else:
        out_dir = Path(__file__).resolve().parent / "videos" / "ui_output" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== インソール設計CLI 実行開始 ===")
    print(f"入力動画: {video}")
    print(f"出力先  : {out_dir}")

    # 1) 2D推定
    blaze = Path(__file__).resolve().parent / "blaze_pose_video.py"
    run(f'python "{blaze}" "{video}" "{out_dir}"')

    # 2) 指標/プロット
    metrics_py = Path(__file__).resolve().parent / "gait_analysis" / "gait_metrics_calc.py"
    run(f'python "{metrics_py}" "{out_dir}"')

    print("=== 完了 ===")
    print(f"- angle_data.csv : {out_dir / 'angle_data.csv'}")
    print(f"- gait_metrics.csv: {out_dir / 'gait_metrics.csv'}")
    print(f"- plots/          : {out_dir / 'plots'}")

if __name__ == "__main__":
    main()
