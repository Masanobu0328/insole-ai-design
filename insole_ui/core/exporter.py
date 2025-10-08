
import cadquery as cq
from pathlib import Path

def export_stl(solid, path: Path):
    cq.exporters.export(solid, str(path))

def export_step(solid, path: Path):
    cq.exporters.export(solid, str(path))
