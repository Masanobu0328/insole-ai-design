# -*- coding: utf-8 -*-
# Fusion 360 script: JSON(+outline) -> STL
import adsk.core, adsk.fusion, traceback, json, os

ARGS = {
    "json": "samples/effective_params_01.json",
    "outline": "samples/foot_outline_demo.csv",
    "out": "out/insole_fusion.stl"
}

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

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

def run(context):
    ui=None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
            design = adsk.fusion.Design.cast(doc.products.itemByProductType('DesignProductType'))
        root = design.rootComponent

        params = load_json(ARGS["json"])
        foot = params.get("foot", {})
        length = float(foot.get("length",260.0))
        width  = float(foot.get("width",100.0))
        side   = foot.get("side","right")
        sgn = 1.0 if side.lower()=="right" else -1.0

        sk = root.sketches.add(root.xYConstructionPlane)
        out_pts = load_outline_csv(ARGS["outline"])
        pts = [adsk.core.Point3D.create(x*length, sgn*y*width, 0) for x,y in out_pts]
        sk.sketchCurves.sketchFittedSplines.add(pts)
        prof = sk.profiles.item(0)

        thk = float(params.get("thickness",{}).get("base",3.0))
        extInput = root.features.extrudeFeatures.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        extInput.setDistanceExtent(False, adsk.core.ValueInput.createByReal(thk))
        body = root.features.extrudeFeatures.add(extInput).bodies.item(0)

        # local mods
        for m in params.get("local_mods", []):
            typ = m.get("type","add")
            amt = float(m.get("amount",0.0))
            rad = float(m.get("radius",10.0))
            x   = float(m.get("x_from_heel",0.0))
            y   = float(m.get("y_from_center",0.0)) * sgn
            sk2 = root.sketches.add(root.xYConstructionPlane)
            sk2.sketchCurves.sketchCircles.addByCenterRadius(adsk.core.Point3D.create(x,y,0), rad)
            prof2 = sk2.profiles.item(0)
            mod = root.features.extrudeFeatures.addSimple(prof2, adsk.core.ValueInput.createByReal(amt),
                                                          adsk.fusion.FeatureOperations.NewBodyFeatureOperation).bodies.item(0)
            tools = adsk.core.ObjectCollection.create(); tools.add(mod)
            comb = root.features.combineFeatures.add(body, tools, typ=="add")
            body = comb.bodies.item(0)

        # heel cup
        depth = float(params.get("heel_cup",{}).get("depth",0.0))
        if depth>0:
            radius = max(18.0, min(32.0, depth*1.4))
            tempOcc = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
            tempComp = adsk.fusion.Component.cast(tempOcc.component)
            spheres = tempComp.features.revolveFeatures
            sk3 = tempComp.sketches.add(tempComp.xYConstructionPlane)
            center = adsk.core.Point3D.create(25+radius,0,0)
            sk3.sketchCurves.sketchCircles.addByCenterRadius(center, radius)
            axis = tempComp.zConstructionAxis
            cup = spheres.addSimple(sk3.profiles.item(0), axis, adsk.core.ValueInput.createByString('360 deg'),
                                    adsk.fusion.FeatureOperations.NewBodyFeatureOperation).bodies.item(0)
            tools = adsk.core.ObjectCollection.create(); tools.add(cup)
            body = root.features.combineFeatures.add(body, tools, False).bodies.item(0)

        # export STL
        expMgr = root.meshManager
        meshexp = adsk.fusion.MeshExportOptions.create(body, os.path.abspath(ARGS["out"]))
        expMgr.export(meshexp)
        if ui: ui.messageBox("Exported: "+os.path.abspath(ARGS["out"]))
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
