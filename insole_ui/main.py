from pathlib import Path
from core.extract_outline_from_image import extract_outline_to_csv
from core.build_insole_cq import build_insole_from_csv
from core.exporter import export_stl, export_step
from core.validator import validate_solid

def run_pipeline(image_path: Path, json_path: Path, side: str, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 中間CSVファイルを作成
    csv_path = out_dir / "outline.csv"
    extract_outline_to_csv(image_path, json_path, csv_path)

    # ここで厚みを定義（必須！）
    thickness = 5.0  # mm

    # インソール生成
    solid = build_insole_from_csv(
        csv_path=csv_path,
        side=side,
        thickness=thickness
    )

    # エラーチェック
    validate_solid(solid)

    # 書き出し
    stl_path = out_dir / f"insole_{side}.stl"
    step_path = out_dir / f"insole_{side}.step"
    export_stl(solid, stl_path)
    export_step(solid, step_path)

    return str(stl_path), str(step_path), str(csv_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python main.py <image_path> <json_path> <left|right> <output_dir>")
        raise SystemExit(1)

    image_path, json_path, side, out_dir = map(Path, sys.argv[1:5])
    run_pipeline(image_path, json_path, side, out_dir)
    print("Done.")
