
import numpy as np

def _interp_profile(xs, anchors):
    pos = np.array([p for p,_ in anchors], dtype=float)
    val = np.array([v for _,v in anchors], dtype=float)
    return np.interp(xs, pos, val)

def med_lat_profiles(xs, arch):
    med_anchors = [
        (0.18, float(arch.get('sustentaculum', 0.0))),
        (0.34, float(arch.get('navicular', 0.0))),
        (0.60, float(arch.get('met1', 0.0))),
        (0.72, float(arch.get('transverse', 0.0))),
    ]
    lat_anchors = [
        (0.28, float(arch.get('cuboid', 0.0))),
        (0.72, float(arch.get('transverse', 0.0))),
    ]
    h_med = _interp_profile(xs, med_anchors)
    h_lat = _interp_profile(xs, lat_anchors)
    return h_med, h_lat

def edge_falloff(ys, half_w, power=6.0):
    t = np.clip((ys + half_w) / (2*half_w), 0, 1)
    base = 0.5 - 0.5*np.cos(np.pi * t)
    return np.power(base, power)

def width_weight(ys, half_w, side):
    w = np.clip((ys + half_w) / (2*half_w), 0, 1)
    if side.lower().startswith("l"):
        w = 1.0 - w
    return w

def clamp_grad(prev, cur, max_delta):
    delta = cur - prev
    delta = np.clip(delta, -max_delta, max_delta)
    return prev + delta
