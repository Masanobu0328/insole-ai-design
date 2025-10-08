#!/usr/bin/env python
# -*- coding: utf-8 -*-
# usage:
#   python tools/excel_to_json.py --excel samples/params.xlsx --outdir samples/from_excel
import os, json, argparse
import pandas as pd

def unflatten(d):
    out = {}
    for k, v in d.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    xl = pd.ExcelFile(args.excel)
    cases = pd.read_excel(xl, "cases")
    lmods = pd.read_excel(xl, "local_mods") if "local_mods" in xl.sheet_names else pd.DataFrame()

    # group local_mods by id
    lmods_by_id = {}
    if not lmods.empty:
        for _, r in lmods.iterrows():
            rid = str(r.get("id","")).strip()
            if not rid: continue
            lmods_by_id.setdefault(rid, []).append({
                "name": r.get("name",""),
                "type": r.get("type","add"),
                "amount": float(r.get("amount",0) or 0),
                "x_from_heel": float(r.get("x_from_heel",0) or 0),
                "y_from_center": float(r.get("y_from_center",0) or 0),
                "radius": float(r.get("radius",0) or 0),
            })

    for _, row in cases.iterrows():
        rid = str(row.get("id","")).strip() or "case"
        flat = {}
        for col, val in row.items():
            if col=="id": continue
            if pd.isna(val): continue
            if isinstance(val, str):
                v = val.strip()
                if v == "": continue
                try:
                    flat[col] = float(v)
                except:
                    flat[col] = v
            else:
                flat[col] = float(val) if isinstance(val,(int,float)) else val
        j = unflatten(flat)
        j.setdefault("meta", {"version":"1.0.0","unit":"mm"})
        j.setdefault("foot", {}); j.setdefault("thickness", {})
        j.setdefault("heel_cup", {}); j.setdefault("arch", {})
        j.setdefault("manufacturing", {})
        j["id"] = rid
        j["local_mods"] = lmods_by_id.get(rid, [])
        out_path = os.path.join(args.outdir, f"effective_params_{rid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        print("Wrote", out_path)

if __name__ == "__main__":
    main()
