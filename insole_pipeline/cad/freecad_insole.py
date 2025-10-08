#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FreeCAD CLI: JSON + outline CSV -> STL
import sys, argparse, json, os
import FreeCAD as App
import Part, Mesh

def load_outline_csv(path):
    pts=[]
    with open(path,'r',encoding='utf-8') as f:
        header=f.readline()
        for line in f:
            line=line.strip()
            if not line: continue
            x,y = line.split(',')
            pts.append((float(x), float(y)))
    return pts

def outline_to_mm(pts, length_mm, width_mm, side="right"):
    sgn = 1.0 if str(side).lower()=="right" else -1.0
    mm=[]
    for x,y in pts:
        X = x*float(length_mm)
        Y = sgn*y*float(width_mm)
        mm.append(App.Vector(X,Y,0))
    if (mm[0]-mm[-1]).Length > 1e-6:
        mm.append(mm[0])
    return mm

def make_face(vectors):
    wire = Part.makePolygon(vectors)
    return Part.Face(wire)

def extrude(face, thk):
    return face.extrude(App.Vector(0,0,thk))

def add_local_mods(solid, params):
    mods = params.get("local_mods", [])
    result = solid
    topZ = result.BoundBox.ZMax
    for m in mods:
        typ = m.get("type","add")
        amt = float(m.get("amount",0.0))
        rad = float(m.get("radius",10.0))
        x   = float(m.get("x_from_heel",0.0))
        y   = float(m.get("y_from_center",0.0))
        cyl = Part.makeCylinder(rad, amt, App.Vector(x,y,topZ))
        result = result.fuse(cyl) if typ=="add" else result.cut(cyl)
        result = result.removeSplitter()
    return result

def add_medial_arch(solid, params):
    arch = params.get("arch", {})
    h = float(arch.get("medial_arch_height",0.0))
    if h<=0: return solid
    x0 = float(arch.get("arch_position_from_heel",120.0))
    w  = float(arch.get("arch_width",45.0))
    segs=5
    rad0=w*0.5
    res=solid
    topZ=res.BoundBox.ZMax
    for i in range(segs):
        cx = x0 + (i - segs/2.0)*6.0
        cy = (i - segs/2.0)*1.5
        bump = Part.makeCylinder(rad0*(0.95**abs(i)), h/segs, App.Vector(cx,cy,topZ))
        res = res.fuse(bump).removeSplitter()
    return res

def carve_heel_cup(solid, params):
    hc = params.get("heel_cup", {})
    depth = float(hc.get("depth",0.0))
    if depth<=0: return solid
    bb = solid.BoundBox
    radius = max(18.0, min(32.0, depth*1.4))
    cup_center = App.Vector(25.0, 0.0, bb.ZMax - depth*0.6)
    sphere = Part.makeSphere(radius, cup_center)
    res = solid.cut(sphere).removeSplitter()
    return res

def refine_and_mesh(solid, out_path):
    if solid.isNull() or not solid.isValid():
        raise RuntimeError("Invalid BRep.")
    mesh = Mesh.Mesh()
    pts, facets = solid.tessellate(0.5)
    for f in facets:
        a,b,c = f
        v1=pts[a]; v2=pts[b]; v3=pts[c]
        mesh.addFacet(v1, v2, v3)
    mesh.write(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--outline", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json,"r",encoding="utf-8") as f:
        params = json.load(f)
    foot = params.get("foot", {})
    length = float(foot.get("length",260.0))
    width  = float(foot.get("width",100.0))
    side   = foot.get("side","right")
    thk = float(params.get("thickness",{}).get("base",3.0))

    pts_norm = load_outline_csv(args.outline)
    pts_mm   = outline_to_mm(pts_norm, length, width, side)
    face = make_face(pts_mm)
    base = extrude(face, thk)

    body = add_medial_arch(base, params)
    body = add_local_mods(body, params)
    body = carve_heel_cup(body, params)
    body = body.removeSplitter()

    out_path = os.path.abspath(args.out)
    refine_and_mesh(body, out_path)
    print("Exported:", out_path)

if __name__ == "__main__":
    main()
