
import streamlit as st
from pathlib import Path
import base64
from main import run_pipeline

st.set_page_config(page_title="Insole UI", layout="wide")
st.title("Insole UI – 写真 + JSON → STL 自動生成")

col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("中敷き写真（.jpg / .png）", type=["jpg","jpeg","png"])
    json_file = st.file_uploader("設計JSON（.json）", type=["json"])
    side = st.selectbox("足の側", ["right", "left"], index=0)
    run_btn = st.button("生成する")

with col2:
    st.markdown("### 使い方")
    st.markdown('''
1. 中敷きの写真を真上から撮影（背景は白やA4用紙推奨）  
2. 設計JSONに足長(mm)・足幅(mm)・アーチ補正値を入力  
3. 「生成する」を押すと、STL/STEPが出力されます  
''')

data_dir = Path("data")
out_dir = Path("output")
out_dir.mkdir(exist_ok=True, parents=True)

if run_btn:
    if not img_file or not json_file:
        st.error("画像とJSONの両方をアップロードしてください。")
        st.stop()

    img_path = data_dir / img_file.name
    json_path = data_dir / json_file.name
    img_path.write_bytes(img_file.read())
    json_path.write_bytes(json_file.read())

    st.info("生成を開始しました…")
    try:
        stl_path, step_path, outline_csv = run_pipeline(img_path, json_path, side, out_dir)
    except Exception as e:
        st.error(f"生成に失敗しました: {e}")
        st.stop()

    st.success("生成に成功しました！")
    st.write("出力ファイル:")
    st.write(f"- STL: `{stl_path}`")
    st.write(f"- STEP: `{step_path}`")
    st.write(f"- Outline CSV: `{outline_csv}`")

    # ダウンロードボタン
    st.download_button("⬇STL ダウンロード", Path(stl_path).read_bytes(), file_name=Path(stl_path).name)
    st.download_button("⬇STEP ダウンロード", Path(step_path).read_bytes(), file_name=Path(step_path).name)

    # 3Dプレビュー（model-viewer / data URI）
    st.markdown("### プレビュー（ベータ）")
    try:
        b64 = base64.b64encode(Path(stl_path).read_bytes()).decode("ascii")
        st.components.v1.html(f'''
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <model-viewer src="data:model/stl;base64,{b64}" alt="Insole Preview" auto-rotate camera-controls style="width:100%; height:520px;"></model-viewer>
        ''', height=540)
    except Exception as e:
        st.warning(f"プレビューに失敗しました: {e}")
