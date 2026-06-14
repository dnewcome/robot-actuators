"""
Involute gear geometry for the planetary input stage.

- External gears (sun, planets) come from bd_warehouse's validated `SpurGear`.
- The internal RING gear has no library generator, so we build its involute tooth
  profile directly: internal teeth point inward (tip at r-addendum, root at
  r+dedendum) and the tooth WIDENS with radius (the opposite of an external tooth).

The math is the standard involute:
    inv(α)      = tan α − α
    invang(ρ)   = √((ρ/r_b)² − 1) − acos(r_b/ρ)      (polar roll angle at radius ρ)
    tooth half-angle (internal) = π/(2z) − inv(α) + invang(ρ)
"""

from math import sqrt, acos, cos, sin, tan, pi, radians

from build123d import (
    BuildPart, BuildSketch, BuildLine, Polyline, Circle, make_face, extrude,
    Hole, Mode,
)
from bd_warehouse.gear import SpurGear


def _inv(a: float) -> float:
    return tan(a) - a


def _invang(rho: float, rb: float) -> float:
    if rho <= rb:
        return 0.0
    return sqrt((rho / rb) ** 2 - 1.0) - acos(rb / rho)


def spur_gear(module: float, teeth: int, thickness: float, bore: float = 0.0,
              pressure_angle: float = 20.0):
    """External involute spur gear (sun / planet) with an optional center bore."""
    with BuildPart() as g:
        SpurGear(module=module, tooth_count=teeth, thickness=thickness,
                 pressure_angle=pressure_angle)
        if bore > 0:
            Hole(radius=bore / 2.0)
    return g.part


def ring_gear_profile(module: float, teeth: int, pressure_angle: float = 20.0,
                      addendum: float = None, dedendum: float = None,
                      flank_steps: int = 6, arc_steps: int = 4):
    """Closed inner-boundary polygon of an internal (ring) gear, CCW.
    Teeth point inward; between teeth the boundary opens out to the root circle."""
    m = module
    z = teeth
    a = addendum if addendum is not None else m
    d = dedendum if dedendum is not None else 1.25 * m
    alpha = radians(pressure_angle)
    r = m * z / 2.0
    rb = r * cos(alpha)
    r_tip = max(r - a, rb + 1e-4)     # innermost; keep above the base circle
    r_root = r + d                    # outermost (space between teeth)
    inv_a = _inv(alpha)

    def tha(rho: float) -> float:     # internal tooth half-angle at radius rho
        return pi / (2 * z) - inv_a + _invang(rho, rb)

    radii_out = [r_tip + (r_root - r_tip) * k / flank_steps for k in range(flank_steps + 1)]
    pts = []
    half_pitch = pi / z
    for i in range(z):
        phi = 2 * pi * i / z
        # leading root arc: sector start -> right flank base
        a0 = phi - half_pitch
        a1 = phi - tha(r_root)
        for k in range(arc_steps):
            ang = a0 + (a1 - a0) * k / arc_steps
            pts.append((r_root * cos(ang), r_root * sin(ang)))
        # right flank: root -> tip (decreasing radius, angle -> phi)
        for rho in reversed(radii_out):
            ang = phi - tha(rho)
            pts.append((rho * cos(ang), rho * sin(ang)))
        # tip land arc across the tooth tip
        ta = tha(r_tip)
        for k in range(arc_steps + 1):
            ang = phi - ta + 2 * ta * k / arc_steps
            pts.append((r_tip * cos(ang), r_tip * sin(ang)))
        # left flank: tip -> root (increasing radius)
        for rho in radii_out:
            ang = phi + tha(rho)
            pts.append((rho * cos(ang), rho * sin(ang)))
        # trailing root arc: left flank base -> sector end
        a2 = phi + tha(r_root)
        a3 = phi + half_pitch
        for k in range(1, arc_steps + 1):
            ang = a2 + (a3 - a2) * k / arc_steps
            pts.append((r_root * cos(ang), r_root * sin(ang)))
    return pts, r_root


def ring_gear(module: float, teeth: int, thickness: float, rim: float = 2.5,
              pressure_angle: float = 20.0):
    """Internal ring gear: an annulus whose bore carries inward-pointing involute teeth.
    OD = root circle + rim. Returns a single watertight solid."""
    pts, r_root = ring_gear_profile(module, teeth, pressure_angle)
    # drop consecutive duplicate points (segment junctions coincide -> zero-length edges)
    clean = [pts[0]]
    for x, y in pts[1:]:
        px, py = clean[-1]
        if (x - px) ** 2 + (y - py) ** 2 > 1e-9:
            clean.append((x, y))
    if (clean[0][0] - clean[-1][0]) ** 2 + (clean[0][1] - clean[-1][1]) ** 2 < 1e-9:
        clean.pop()
    pts = clean
    with BuildPart() as ring:
        with BuildSketch() as sk:
            Circle(radius=r_root + rim)
            with BuildLine():
                Polyline(*pts, close=True)
            make_face(mode=Mode.SUBTRACT)     # carve the toothed bore
        extrude(amount=thickness)
    return ring.part


if __name__ == "__main__":
    from build123d import export_stl
    sun = spur_gear(0.5, 12, 4.0, bore=3.0)
    planet = spur_gear(0.5, 12, 4.0, bore=1.5)
    ring = ring_gear(0.5, 36, 4.0)
    for name, part in (("sun", sun), ("planet", planet), ("ring_gear", ring)):
        print(f"{name:10s} solids={len(part.solids())} valid={part.is_valid} "
              f"bbox={part.bounding_box().size.X:.2f}mm")
        export_stl(part, f"/tmp/_{name}.stl")
    print("exported /tmp/_*.stl")
