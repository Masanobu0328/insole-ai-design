#!/usr/bin/env python
# -*- coding: utf-8 -*-
# usage:
#   python tools/json_to_excel.py --inputs samples --out samples/params.xlsx
import os, json, argparse, glob
import pandas as pd

def load_jsons(input_path):
    files = []
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.json")))
    else:
        files = [input_path]
    data = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data.append(json.load(f))
    return data

def flatten_dict(d, parent='', out=None):
    if out is None: out = {}
    for k,v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            flatten_dict(v, key, out)
        else:
            out[key] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    json_list = load_jsons(args.inputs)

    case_rows, lm_rows = [], []
    for j in json_list:
        j2 = dict(j)
        local_mods = j2.pop("local_mods", [])
        flat = flatten_dict(j2)
        flat["id"] = j.get("id","")
        case_rows.append(flat)
        for idx, m in enumerate(local_mods):
            row = {"id": j.get("id",""), "index": idx}
            row.update(m)
            lm_rows.append(row)

    cases_df = pd.DataFrame(case_rows).fillna("")
    if "id" in cases_df.columns:
        cols = ["id"] + [c for c in cases_df.columns if c != "id"]
        cases_df = cases_df[cols]
    lmods_df = pd.DataFrame(lm_rows).fillna("") if lm_rows else pd.DataFrame(
        columns=["id","index","name","type","amount","x_from_heel","y_from_center","radius"]
    )

    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        cases_df.to_excel(w, sheet_name="cases", index=False)
        lmods_df.to_excel(w, sheet_name="local_mods", index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
