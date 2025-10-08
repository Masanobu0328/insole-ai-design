import os
import json
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

# ========== GPT関連（安全フォールバック付き） ==========

def _safe_to_series(x):
    try:
        if isinstance(x, (list, tuple)):
            return pd.Series(list(x), dtype="float64")
        if hasattr(x, "values") and hasattr(x, "dropna"):
            try:
                s = pd.to_numeric(x, errors="coerce")
            except Exception:
                s = pd.Series(x)
            return s
        return pd.Series([x], dtype="float64")
    except Exception:
        return x

def robust_range(x) -> float:
    try:
        s = _safe_to_series(x)
        if hasattr(s, "dropna"):
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return 0.0
            return float(np.nanpercentile(s, 95) - np.nanpercentile(s, 5))
        return 0.0
    except Exception:
        return 0.0

def load_interview_for_case(case: str, input_root: Path) -> dict:
    for name in [
        f"{case}_interview.json",
        f"{case}.interview.json",
        "interview.json",
    ]:
        path = input_root / case / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not any(v for v in data.values()):
                print(f"[warn] interview.json が空です: {path}")
            return data
    print(f"[warn] interview.json が見つかりません: {input_root / case}")
    return {}

def build_compact_metrics_dict(metrics_csv_path: Path) -> dict:
    try:
        df = pd.read_csv(metrics_csv_path)
        return build_compact_metrics_dict_from_df(df)
    except Exception:
        return {}

def build_compact_metrics_dict_from_df(df: pd.DataFrame) -> dict:
    d: dict = {}
    try:
        if set(["metric","value"]).issubset(map(str.lower, df.columns)):
            mcol = [c for c in df.columns if c.lower()=="metric"][0]
            vcol = [c for c in df.columns if c.lower()=="value"][0]
            for _, r in df.iterrows():
                key = str(r[mcol]).strip()
                try:
                    val = float(r[vcol])
                except Exception:
                    val = r[vcol]
                d[key] = val
        else:
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                row = num_df.iloc[0].to_dict()
                d.update({str(k): float(v) for k, v in row.items()})

        def alias(src_keys, dst_key):
            for k in src_keys:
                if k in d and dst_key not in d:
                    d[dst_key] = d[k]
                    break

        alias(["pelvis_sway","pelvis_sway_px","pelvis_lateral_sway_px"], "pelvis_sway")
        alias(["shoulder_diff_px","shoulder_height_diff_px"], "shoulder_diff_px")
        alias(["knee_varus_valgus_diff","knee_abb_add_diff_deg"], "knee_varus_valgus_diff")
        alias(["heel_tilt_diff","ankle_inv_evr_diff_deg","ankle_pronation_supination_diff_deg"], "heel_tilt_diff")
    except Exception:
        pass
    return d

try:
    from openai import OpenAI
    _tmp = os.getenv("OPENAI_API_KEY")
    _client: Optional[OpenAI] = OpenAI(api_key=_tmp) if _tmp else None
except Exception:
    _client = None

SYSTEM_THERAPIST = (
    "あなたは理学療法士レベルの歩行解析者です。"
    "与えられた 'interview'（問診）と 'metrics'（数値） だけを根拠に、"
    "臨床的に有用な所見を日本語で作成してください。診断名の断定は禁止。"
    "【出力フォーマット（必須）】\n"
    "1) 主要所見（要点・左右差・大振幅）\n"
    "2) 前額面（XY）: 股関節 内転/外転、膝 内反/外反、足関節 回内/回外\n"
    "3) 矢状面（XZ）: 股関節 屈曲/伸展、膝 屈曲/伸展、足関節 背屈/底屈\n"
    "4) 水平面（YZ, 近似）: 股関節・膝 内旋/外旋（可能な範囲で）\n"
    "5) 骨盤：前後傾、挙上下制、回旋（可能な範囲で）\n"
    "6) 機能的示唆：どの局面で出やすいか（一般論で可）\n"
    "7) 測定限界：pxは相対、撮影条件依存、短尺時不安定など\n"
    "箇条書き20〜30行、数値は与えられた単位（px/°）のまま。"
)

SYSTEM_PATIENT = (
    "あなたは患者さんにやさしく説明する理学療法士です。"
    "与えられた 'interview'（問診）と 'metrics'（数値）を基に、"
    "専門用語を避け、左右差や全体傾向をわかりやすく短文で説明してください。"
    "筋肉名は避け、『お尻の筋肉』『右の腰の筋肉』など日常表現を使うこと。"
    "10~20文。診断断定は禁止。pxなど専門単位は極力避け、最後に日常の簡単アドバイスを1つ添える。"
)

THERAPIST_PROMPT = (
    "以下は問診と解析サマリです。これを根拠に、上のルールで所見を作成してください。\n"
    "【問診】\n{interview_json}\n\n"
    "【数値サマリ】\n{metrics_json}\n"
)

PATIENT_PROMPT = (
    "以下は問診と解析サマリです。これを参考に、上のルールで患者さん向けメッセージを作成してください。\n"
    "【問診】\n{interview_json}\n\n"
    "【数値サマリ】\n{metrics_json}\n"
)

def call_openai_o4(system_prompt: str, user_prompt: str) -> str:
    if _client is None:
        return "[AI生成エラー] APIキー未設定またはクライアント初期化失敗"

    try:
        resp = _client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else "[AI生成エラー] 空レスポンス"
    except Exception as e:
        return f"[AI生成エラー] {e}"

def generate_patient_review(metrics: Union[Path, pd.DataFrame], interview: dict) -> str:
    if isinstance(metrics, pd.DataFrame):
        data = build_compact_metrics_dict_from_df(metrics)
    else:
        data = build_compact_metrics_dict(metrics)

    prompt = PATIENT_PROMPT.format(
        interview_json=json.dumps(interview, ensure_ascii=False, indent=2),
        metrics_json=json.dumps(data, ensure_ascii=False, indent=2),
    )

    print("=== GPT送信プロンプト ===")
    print(prompt)

    txt = call_openai_o4(SYSTEM_PATIENT, prompt)

    print("=== GPT返答 ===")
    print(txt)

    if txt.startswith("[AI生成エラー]"):
        lines = ["【自動レビュー（簡易）】"]
        if "pelvis_sway" in data:
            lines.append(f"- 骨盤の横揺れはやや{'大きめ' if float(data['pelvis_sway'])>80 else '小さめ〜普通'}に見えます。")
        if "knee_varus_valgus_diff" in data:
            lines.append(f"- 膝の左右差（内側/外側への傾き）は目安で {data['knee_varus_valgus_diff']}°。")
        if "heel_tilt_diff" in data:
            lines.append(f"- かかとの左右差は目安で {data['heel_tilt_diff']}°。")
        lines.append("→ 無理のない範囲で、お尻まわり・体幹をやさしく動かす/ほぐすことから始めましょう。これは診断ではありません。")
        return "\n".join(lines)
    return txt

def generate_therapist_review(metrics: Union[Path, pd.DataFrame], interview: dict) -> str:
    if isinstance(metrics, pd.DataFrame):
        data = build_compact_metrics_dict_from_df(metrics)
    else:
        data = build_compact_metrics_dict(metrics)

    prompt = THERAPIST_PROMPT.format(
        interview_json=json.dumps(interview, ensure_ascii=False, indent=2),
        metrics_json=json.dumps(data, ensure_ascii=False, indent=2),
    )
    txt = call_openai_o4(SYSTEM_THERAPIST, prompt)
    if txt.startswith("[AI生成エラー]"):
        return (
            "【所見（簡易フォールバック）】\n"
            f"- pelvis sway(px): {data.get('pelvis_sway','-')}, "
            f"knee varus/valgus diff(°): {data.get('knee_varus_valgus_diff','-')}, "
            f"heel tilt diff(°): {data.get('heel_tilt_diff','-')}\n"
            "→ 前額面は股外側筋群のスタビリティ、矢状面は足関節背屈の可動域、水平面は骨盤回旋コントロールを優先。\n"
            "（診断ではなく運動学的傾向の所見）"
        )
    return txt
