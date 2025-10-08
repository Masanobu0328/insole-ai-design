# -*- coding: utf-8 -*-
"""
本番用 analyze_all.py（問診統合 / GPTレビュー内蔵 / ui_output集約）
入力:  videos/ui_input/<case>_front.mp4, <case>_side.mp4
出力:  videos/ui_output/<case>_merged/ に集約

生成物:
  - angle_data.csv        (基礎データ)
  - gait_metrics.csv      (まとめ)
  - plots/                (可視化)
  - review_patient.txt    (患者向けレビュー)
  - review_therapist.txt  (セラピスト向けレビュー)

■問診の読み取りルール（どれか一つでもあれば使用）
  videos/ui_input/
    ├─ <case>_interview.json
    ├─ <case>.interview.json
    └─ <case>/interview.json

JSONの例:
{
  "主訴": "右膝の違和感",
  "違和感の部位": "右膝",
  "生活習慣": "座位作業が多い",
  "運動習慣": "ウォーキング週2",
  "既往歴": "足関節捻挫"
}

※APIキーは環境変数 OPENAI_API_KEY に設定してください。
  setx OPENAI_API_KEY "sk-xxxx"
"""
import argparse
import re
import sys
import os
import json
from pathlib import Path
import subprocess
import shlex
import traceback
from typing import Optional
import pandas as pd
from streamlit.gpt_review_utils import (
    load_interview_for_case,
    generate_patient_review,
    generate_therapist_review,
)
import shutil
# ========== ベース設定 ==========
BASE_DIR = Path(__file__).resolve().parent

# ========== ユーティリティ ==========

def run(cmd: str):
    """子プロセスの出力をリアルタイムで表示"""
    print(">>", cmd, flush=True)
    if cmd.strip().startswith("python "):
        py = sys.executable or "python"
        cmd = cmd.replace("python ", f'"{py}" -u ', 1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)  # gait_analysis を -m で呼べるように
    ret = subprocess.call(shlex.split(cmd), env=env, cwd=str(BASE_DIR))
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd} (exit={ret})")

def stem_case_role(path: Path):
    """<case>_front / <case>_side を抽出"""
    m = re.match(r"^(?P<case>.+)_(?P<role>front|side)$", path.stem, flags=re.IGNORECASE)
    if m:
        return m.group("case"), m.group("role").lower()
    m2 = re.match(r"^(?P<case>.+)_(?P<role>front|side)\.(mp4|mov|avi|mkv)$", path.name, flags=re.IGNORECASE)
    if m2:
        return m2.group("case"), m2.group("role").lower()
    return None, None

def discover_pairs_from_videos(input_root: Path):
    """ui_input 内の *_front.* と *_side.* をペアリング"""
    vids = []
    for p in input_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".mp4",".mov",".mkv",".avi"}:
            case, role = stem_case_role(p)
            if case and role:
                vids.append((case, role, p))
    pairs = {}
    for case, role, p in vids:
        pairs.setdefault(case, {})[role] = p
    return {case: d for case, d in pairs.items() if "front" in d and "side" in d}

def have_angle_csv(dir_path: Path) -> bool:
    return (dir_path / "angle_data.csv").exists()

def have_metrics(dir_path: Path) -> bool:
    return (dir_path / "gait_metrics.csv").exists() and (dir_path / "plots").exists()

# ========== パイプライン各段階 ==========

def ensure_single_video_processed(video_path: Path, output_root: Path, force: bool) -> Path:
    """
    動画1本を main.py で処理（出力先は ui_output/<stem>/）
    既に angle_data.csv があればスキップ（--force で再実行）
    """
    out_dir = output_root / video_path.stem
    need_run = force or not have_angle_csv(out_dir)
    if need_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        main_py = BASE_DIR / "main.py"
        # 出力先を第2引数で明示
        run(f'python "{main_py}" "{video_path}" "{out_dir}"')
    else:
        print(f"[skip] already processed: {out_dir}")
    return out_dir

def merge_and_metrics(front_dir: Path, side_dir: Path, output_root: Path, case: str, force: bool, interview: dict, input_root: Path)-> Path:
    """
    front/side 出力フォルダをマージ → gait_metrics を統合フォルダに作成 → GPTレビュー保存
    戻り値: merged_dir Path
    """
    merged_dir = output_root / f"{case}_merged"
    
     # ✅ interview.json を case フォルダにコピー（GPTレビューが正しく動くために必要）
    try:
        shutil.copy(input_root / "interview.json", input_root / case / "interview.json")
    except Exception as e:
        print(f"[warn] interview.json のコピーに失敗しました: {e}")
    # --- マージ ---
    need_merge = force or not have_angle_csv(merged_dir)
    if need_merge:
        merged_dir.mkdir(parents=True, exist_ok=True)
        run(
            f'python -m gait_analysis.merge_two_views '
            f'--front_dir "{front_dir}" --side_dir "{side_dir}" --out_dir "{merged_dir}"'
        )
    else:
        print(f"[skip] already merged: {merged_dir}")

    # --- メトリクス ---
    need_metrics = force or not have_metrics(merged_dir)
    if need_metrics:
        run(f'python -m gait_analysis.gait_metrics_calc "{merged_dir}"')
    else:
        print(f"[skip] metrics exist: {merged_dir}")

    # --- AIレビュー生成（gait_metrics.csv があるときのみ） ---
    metrics_csv = merged_dir / "gait_metrics.csv" 
    if metrics_csv.exists():
        try:
            metrics_df = pd.read_csv(metrics_csv)

            if metrics_df.empty:
                print(f"[warn] gait_metrics.csv は存在するが中身が空: {metrics_csv}")
                return merged_dir  # もしくは何もせず終了してよい

            interview_dict = load_interview_for_case(case, input_root)

            if not interview_dict or not any(interview_dict.values()):
                print(f"[warn] interview.json が空です: {input_root / case / 'interview.json'}")

            patient_txt = generate_patient_review(metrics_df, interview_dict)
            therapist_txt = generate_therapist_review(metrics_df, interview_dict)

            (merged_dir / "review_patient.txt").write_text(patient_txt, encoding="utf-8")
            (merged_dir / "review_therapist.txt").write_text(therapist_txt, encoding="utf-8")
        except Exception as e:
            print(f"[warn] GPTレビュー生成で例外: {e}")
    else:
        print(f"[warn] gait_metrics.csv が見つかりません: {metrics_csv}")
    
# ========== メイン ==========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_root",
        type=str,
        default="streamlit/videos/ui_input",   # 本番: ui_input
        help="入力動画の親フォルダ (省略時: videos/ui_input)"
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default="streamlit/videos/ui_output",  # 本番: ui_output
        help="出力の親フォルダ (省略時: videos/ui_output)"
    )
    ap.add_argument("--force", action="store_true", help="既存結果があっても上書き実行")
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"[!] input_root が見つかりません: {input_root}")
        sys.exit(1)

    pairs = discover_pairs_from_videos(input_root)
    if not pairs:
        print(f"[!] {input_root} に *_front と *_side のペア動画が見つかりません。")
        sys.exit(1)

    print(f"検出したペア数: {len(pairs)}\n")

    ok, ng = 0, 0
    try:
        for case, d in sorted(pairs.items()):
            print(f"=== Case: {case} ===")
            try:
                # 1) 問診をロード（なければ {}）
                interview = load_interview_for_case(case, input_root)

                # 2) front/side を main.py で処理
                front_out = ensure_single_video_processed(d["front"], output_root, args.force)
                side_out  = ensure_single_video_processed(d["side"],  output_root, args.force)

                # 3) 統合 → 指標/可視化 → GPTレビュー保存
                merged_dir = merge_and_metrics(front_out, side_out, output_root, case, args.force, interview, input_root.parent)
                print(f"  OK merged: {merged_dir}\n")
                ok += 1
            except Exception as e:
                print(f"  ERROR in {case}: {e}")
                traceback.print_exc()
                ng += 1
    finally:
        print("==== サマリ ====")
        print(f"成功: {ok} / 失敗: {ng}")

if __name__ == "__main__":
    main()