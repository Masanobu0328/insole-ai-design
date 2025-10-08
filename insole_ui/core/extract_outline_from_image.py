# extract_outline_from_image.py（改善版）

import cv2, numpy as np, json, pandas as pd
from pathlib import Path
from scipy.interpolate import splprep, splev

def extract_outline_to_csv(image_path: Path, json_path: Path, save_csv: Path, n_points: int = 120):
    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    length_mm = float(params.get('foot', {}).get('length', 260.0))

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")

    print(f"[INFO] 画像サイズ: {img.shape}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # debug: 輪郭抽出マスク保存
    cv2.imwrite("debug_mask.jpg", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("輪郭が抽出できませんでした（背景やコントラストを確認してください）。")

    c = max(contours, key=cv2.contourArea)
    pts = np.squeeze(c)

    if pts.ndim != 2 or pts.shape[0] < 5:
        raise RuntimeError(f"抽出輪郭が不十分: {pts.shape[0]} 点しか見つかりませんでした。画像の背景・照明・コントラストをご確認ください。")

    # ---- スプライン補間 ----
    pts = pts.T
    tck, u = splprep(pts, s=5.0, per=True)
    u_fine = np.linspace(0, 1, n_points)
    outline_interp = splev(u_fine, tck)
    outline_interp = np.vstack(outline_interp).T

    # ---- スケーリング（mm単位） ----
    x_min, x_max = float(np.min(outline_interp[:,0])), float(np.max(outline_interp[:,0]))
    px_len = max(x_max - x_min, 1e-6)
    scale_mm_per_px = length_mm / px_len
    y_center = float(np.mean(outline_interp[:,1]))

    outline_mm = np.column_stack([outline_interp[:,0] - x_min, outline_interp[:,1] - y_center]) * scale_mm_per_px

    # ---- 閉ループ処理 ----
    if not np.allclose(outline_mm[0], outline_mm[-1]):
        outline_mm = np.vstack([outline_mm, outline_mm[0]])

    # ---- 保存 ----
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(outline_mm, columns=['x_mm','y_mm'])
    df.to_csv(save_csv, index=False)
    return save_csv
