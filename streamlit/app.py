# -*- coding: utf-8 -*-
"""
Streamlit UI: 動画アップロード → analyze_all.py 実行 → ui_output/<case>_merged を読み込み
"""

from __future__ import annotations
import os, sys, json, re, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from gpt_review_utils import generate_patient_review, generate_therapist_review
import sys
from subprocess import run

# =============== パス設定 ===============
ST_DIR   = Path(__file__).resolve().parent
ROOT_DIR = ST_DIR.parent
ANALYZE  = ROOT_DIR / "analyze_all.py"

UI_INPUT_DIR  = ST_DIR / "videos" / "ui_input"
UI_OUTPUT_DIR = ST_DIR / "videos" / "ui_output"
UI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
UI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============== OpenAI クライアント ===============
try:
    from openai import OpenAI
    _tmp = os.getenv("OPENAI_API_KEY")
    _client: Optional[OpenAI] = OpenAI(api_key=_tmp) if _tmp else None
except Exception:
    _client = None

# =============== ユーティリティ ===============
def run_subprocess(cmd_list, cwd: Path) -> Tuple[bool, str]:
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR)
        res = subprocess.run(cmd_list, cwd=str(cwd), env=env, text=True, capture_output=True)
        out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        return res.returncode == 0, out
    except Exception as e:
        return False, f"[LauncherError] {e}"

def sanitize_for_fs(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("._") or "case"

# =============== GPT関連（省略: あなたの定義そのまま） ===============
# call_openai_o4, generate_patient_review, generate_therapist_review などはそのまま使ってください

# =============== Streamlit UI ===============
st.set_page_config(page_title="歩行解析レビュー", layout="wide")
st.title("歩行解析レビュー")

if "step" not in st.session_state:
    st.session_state["step"] = 1

# --- Step 1: 基本情報 ---
if st.session_state["step"] == 1:
    st.header("ステップ 1: 基本情報")
    st.text_input("お名前", key="name", value=st.session_state.get("name", ""))
    st.number_input("年齢", 0, 120, key="age", value=st.session_state.get("age", 0))
    st.selectbox("性別", ["未選択", "男性", "女性", "その他"], key="sex", index=["未選択", "男性", "女性", "その他"].index(st.session_state.get("sex", "未選択")))
    st.number_input("身長(cm)", 0, 250, key="height", value=st.session_state.get("height", 0))
    st.number_input("体重(kg)", 0, 250, key="weight", value=st.session_state.get("weight", 0))

    if st.button("次へ"):
    #明示的に session_state に保存
       st.session_state["step"] = 2
       st.rerun()

# --- Step 2: 主訴と目標 ---
elif st.session_state["step"] == 2:
    st.header("ステップ 2: 主訴と目標")
    st.text_area("主訴/気になること", key="chief", value=st.session_state.get("chief", ""))
    st.text_area("目標（いつまでに何を）", key="goal", value=st.session_state.get("goal", ""))


    if st.button("戻る"):
        st.session_state["step"] = 1
        st.rerun()
    if st.button("次へ"):
        st.session_state["step"] = 3
        st.rerun()

# --- Step 3: 生活スタイルと仕事・運動 ---
elif st.session_state["step"] == 3:
    st.header("ステップ 3: 生活スタイルと仕事・運動")
    st.text_area("生活スタイル（例: 座っている時間が長い、立ち仕事が多い 等）", key="lifestyle_detail", value=st.session_state.get("lifestyle_detail", ""))
    st.text_input("お仕事", key="job", value=st.session_state.get("job", ""))
    st.text_area("運動習慣（例: ジョギングを週2回 など）", key="exercise", value=st.session_state.get("exercise", ""))


    if st.button("戻る"):
        st.session_state["step"] = 2
        st.rerun()
    if st.button("次へ"):
        st.session_state["step"] = 4
        st.rerun()

# --- Step 4: 動画アップロード & 実行 ---
elif st.session_state["step"] == 4:
    st.header("ステップ 4: 動画アップロード")

    # case の生成（この時点で name が確実に入力済）
    if "case" not in st.session_state:
        name = st.session_state.get("name", "case")
        case_base = sanitize_for_fs(name)
        case = f"{case_base}_{datetime.now().strftime('%Y%m%d_%H%M')}"  # 秒は省略
        st.session_state["case"] = case
    else:
        case = st.session_state["case"]

    # 入力パス
    case_in = UI_INPUT_DIR / case
    case_in.mkdir(parents=True, exist_ok=True)

    # --- UI ---
    f_front = st.file_uploader("正面動画", type=["mp4", "mov", "mkv", "avi"], key="front")
    f_side  = st.file_uploader("側面動画", type=["mp4", "mov", "mkv", "avi"], key="side")

    if "case" not in st.session_state:
         name = st.session_state.get("name", "case").strip()
         case_base = sanitize_for_fs(name)
         case = f"{case_base}_{datetime.now().strftime('%Y%m%d_%H%M')}"
         st.session_state["case"] = case
    else:
         case = st.session_state["case"]

    if st.button("戻る"):
        st.session_state["step"] = 3
        st.rerun()

    if st.button("解析を実行する"):
        if f_front is None or f_side is None:
            st.error("正面と側面の両方の動画をアップロードしてください。")
            st.stop()

        #動画ファイル保存
        with open(case_in / f"{case}_front.mp4", "wb") as f:
            f.write(f_front.read())
        with open(case_in / f"{case}_side.mp4", "wb") as f:
            f.write(f_side.read())

        #問診保存（name, age, sex ...）
        interview = {
            k: st.session_state.get(k, "")
            for k in ["name", "age", "sex", "height", "weight", "chief", "goal", "lifestyle_detail", "job", "exercise"]
        }
        with open(case_in / "interview.json", "w", encoding="utf-8") as f:
            json.dump(interview, f, ensure_ascii=False, indent=2)

        #解析スクリプト呼び出し
        py = sys.executable or "python"
        cmd = [py, str(ANALYZE), "--input_root", str(UI_INPUT_DIR), "--output_root", str(UI_OUTPUT_DIR)]
        ok, log = run_subprocess(cmd, cwd=ROOT_DIR)

        #ログ表示
        st.text_area("解析ログ", value=log, height=180)

        if ok:
            st.success("解析が完了しました。")
            st.session_state["last_case"] = case
            st.rerun()
        else:
            st.error("解析に失敗しました。")

# --- 下段: 結果表示 ---
last_case = st.session_state.get("last_case")
if last_case:
    merged_dir = UI_OUTPUT_DIR / f"{last_case}_merged"
    st.subheader(f"結果: {merged_dir.name}")

    metrics_csv = merged_dir / "gait_metrics.csv"
    plots_dir   = merged_dir / "plots"

    gm_df = pd.read_csv(metrics_csv) if metrics_csv.exists() else None
    if gm_df is not None:
        st.dataframe(gm_df, use_container_width=True)

        try:
            # 問診読み込み
            interview_path = UI_INPUT_DIR / last_case / "interview.json"
            interview = json.load(open(interview_path, encoding="utf-8")) if interview_path.exists() else {}

            # GPTレビュー生成
            patient_txt   = generate_patient_review(gm_df, interview)
            therapist_txt = generate_therapist_review(gm_df, interview)

            tabs = st.tabs(["患者向けレビュー", "セラピスト向け所見", "プロット"])
            with tabs[0]:
                st.subheader("患者向け 自動レビュー")
                st.write(patient_txt)
            with tabs[1]:
                st.subheader("セラピスト向け 所見")
                st.text(therapist_txt)
            with tabs[2]:
                if plots_dir.exists():
                    for p in sorted(plots_dir.glob("*.png")):
                        st.image(str(p), caption=p.name, use_column_width=True)
        except Exception as e:
            st.warning(f"レビュー生成でエラー: {e}")
    else:
        st.warning("gait_metrics.csv が見つかりません。")
