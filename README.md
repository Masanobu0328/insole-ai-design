# Takumi Insole System — SPEC (AI Edition)

## Purpose
AI analyzes gait videos & questionnaire data → generates custom insoles via CadQuery.

## Input / Output
- Input: video_front.mp4, video_side.mp4, questionnaire.json
- Output: gait_metrics.csv, params_final.json, insole_right.stl, insole_left.stl

## Pipeline
1. `analyze_all.py` : extract pose (MoveNet) → angle_data.csv  
2. `gait_metrics_calc.py` : compute speed, asymmetry, pelvic tilt  
3. `param_builder.py` : convert metrics → JSON params  
4. `build_insole_from_json.py` : generate STL via CadQuery  
5. `streamlit/app.py` : visualize + therapist review

## Tech
Python 3.10, TensorFlow Lite, OpenCV, CadQuery 2.4, Streamlit

## Notes
- All paths under `/videos/`  
- Output units: millimeters  
- JSON params follow `data/params/params_schema.json`
