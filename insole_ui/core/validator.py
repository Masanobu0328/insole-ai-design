
def validate_solid(solid):
    v = solid.val()
    if (not v.isValid()) or v.Volume() <= 0:
        raise RuntimeError("Loft結果が不正です（体積0またはBRep不整合）。")
    return True
