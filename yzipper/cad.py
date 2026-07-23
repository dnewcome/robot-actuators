"""
Parametric CAD for the Y-ZIPPER, following the paper's tooth geometry (CHI '26, §3.1).

Corrected from the earlier rectangular-castellation version. Each strip is one face of
the triangular prism, laid out FLAT for printing. Along it runs a row of rigid TEETH on
both long edges, joined by thin compliant BRIDGES. Each tooth is a rounded (wave-profile)
prong carrying a BALL NODE (∅1=2.4) on its top face and a SOCKET (∅2=3.0) on its
underside (§3.1) — the ball/socket give shear/alignment, the teeth give compression
support, the bridges carry tension when zipped.

The BEND flat-pattern is a CURVED arc ribbon with differential tooth spacing between the
two edges — that differential is exactly the accumulated tooth-thickness difference
(T1−T2) that sets the bend angle θ = 2(T1−T2)/(√3·w) in yzipper.py.

Parts (watertight solids):
  strip_straight   flat straight strip: rounded teeth + ball/socket + bridges (uniform)
  strip_bend       curved arc strip: differential tooth spacing → programmed bend
  slider           the 3-way slider collar (Separator over Converger) with pull tab

Frame: strip laid flat — X = longitudinal (teeth repeat), Y = width w (edges at ±w/2),
Z = tooth height H / print-up. mm throughout.

Run:  ../.venv/bin/python yzipper/cad.py
Out:  yzipper/out/*.step, *.stl, assembly.step
"""

import sys
from dataclasses import dataclass
from math import sqrt, sin, cos, radians
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, Box, Cylinder, Sphere, RegularPolygon, extrude,
    Locations, Pos, Rot, Compound, Mode, Align,
    export_step, export_stl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yzipper import YZipperParams

_CTR = (Align.CENTER, Align.CENTER, Align.CENTER)
_YMIN = (Align.CENTER, Align.MIN, Align.CENTER)


@dataclass
class Cad:
    P: YZipperParams

    n_teeth: int = 14                # teeth per edge in the CAD sample
    pitch_mm: float = 6.0            # tooth pitch along the length
    tooth_reach: float = 3.0         # how far a tooth juts past the edge (Y) [mm]
    bridge_D: float = 1.2            # compliant bridge / panel thickness (Z) [mm]

    @property
    def w(self) -> float: return self.P.width_w_mm
    @property
    def t(self) -> float: return max(self.P.tooth_t_mm, 2.2)   # tooth length along X (printable)
    @property
    def H(self) -> float: return self.P.tooth_H_mm             # tooth height (Z)
    @property
    def r_ball(self) -> float: return self.P.ball_d1_mm / 2.0
    @property
    def r_socket(self) -> float: return self.P.socket_d2_mm / 2.0
    @property
    def L(self) -> float: return self.n_teeth * self.pitch_mm


def _tooth(cad: Cad):
    """One rounded (wave) tooth protruding +Y from the edge, ball on top, socket below.
    Returned as an algebra-mode Part centred at the edge (y=0)."""
    t, reach, H = cad.t, cad.tooth_reach, cad.H
    with BuildPart() as tp:
        with Locations(Pos(0, 0, 0)):
            Box(t, reach, H, align=_YMIN)                          # prong body
        with Locations(Pos(0, reach, 0)):
            Cylinder(radius=t / 2.0, height=H, align=_CTR)         # rounded wave cap
        with Locations(Pos(0, reach * 0.62, H / 2.0 - cad.r_ball * 0.35)):
            Sphere(radius=cad.r_ball)                              # ball node (top)
        with Locations(Pos(0, reach * 0.62, -H / 2.0 + cad.r_socket * 0.35)):
            Sphere(radius=cad.r_socket, mode=Mode.SUBTRACT)        # socket (underside)
    return tp.part


def make_strip_straight(cad: Cad):
    """Flat straight strip: panel/bridges + uniform rounded teeth on both edges."""
    L, w, D = cad.L, cad.w, cad.bridge_D
    tooth = _tooth(cad)
    part = Pos(L / 2.0, 0, 0) * Box(L, w, D)                       # panel (bridge membrane)
    for k in range(cad.n_teeth):
        xc = (k + 0.5) * cad.pitch_mm
        part += Pos(xc, w / 2.0, 0) * tooth                        # +Y edge
        xo = (k + 1.0) * cad.pitch_mm
        if xo < L:
            part += Pos(xo, -w / 2.0, 0) * Rot(0, 0, 180) * tooth  # -Y edge (staggered)
    return part


def make_strip_bend(cad: Cad, theta_deg: float = 60.0):
    """Curved arc flat-pattern with DIFFERENTIAL tooth spacing between the two edges.

    The ribbon centreline is an arc; the outer edge's teeth are spaced wider than the
    inner edge's. The accumulated tooth-thickness difference (T1−T2) between the edges is
    what produces the bend angle θ = 2(T1−T2)/(√3·w) once zipped — matching Fig. 6b.
    """
    w, D = cad.w, cad.bridge_D
    theta = radians(theta_deg)
    n = cad.n_teeth
    R = (n * cad.pitch_mm) / theta                                 # centreline arc radius
    tooth = _tooth(cad)
    dphi = theta / n
    part = None
    for k in range(n):
        phi = (k + 0.5) * dphi
        cx, cy = R * sin(phi), R * (1 - cos(phi))
        base = Pos(cx, cy, 0) * Rot(0, 0, -theta_deg * (k + 0.5) / n)
        seg = base * Box(cad.pitch_mm * 1.02, w, D)                # tangent panel segment
        seg += base * Pos(0, w / 2.0, 0) * tooth                   # outer edge tooth
        seg += base * Pos(0, -w / 2.0, 0) * Rot(0, 0, 180) * tooth # inner edge tooth
        part = seg if part is None else part + seg
    return part


def make_slider(cad: Cad):
    """3-way slider: a triangular collar (Converger) with a Separator taper + pull tab."""
    thick = 3.0 * cad.pitch_mm
    r_bore = cad.w / sqrt(3.0) + 0.8
    r_out = r_bore + 3.5
    in_out = r_out / 2.0
    with BuildPart() as part:
        with BuildSketch():
            RegularPolygon(radius=r_out, side_count=3, rotation=90)
        extrude(amount=thick)
        with BuildSketch():
            RegularPolygon(radius=r_bore, side_count=3, rotation=90)
        extrude(amount=thick, mode=Mode.SUBTRACT)
        with Locations(Pos(0, -(in_out + 6.0 - 3.0), thick / 2.0)):
            Box(6.0, 12.0, thick * 0.6, align=_CTR)
    return part.part


def make_segment(cad: Cad):
    """A short closed triangular-prism segment (circumradius w/√3, one pitch long),
    instanced along a programmed centreline in the sim to show the zipped rod."""
    with BuildPart() as part:
        with BuildSketch():
            RegularPolygon(radius=cad.w / sqrt(3.0), side_count=3, rotation=90)
        extrude(amount=cad.pitch_mm)
    return part.part


def build_all(cad: Cad) -> dict:
    return {
        "strip_straight": make_strip_straight(cad),
        "strip_bend": make_strip_bend(cad),
        "slider": make_slider(cad),
        "segment": make_segment(cad),
    }


def assembly(cad: Cad) -> Compound:
    s = make_strip_straight(cad)
    b = make_strip_bend(cad)
    sl = make_slider(cad)
    return Compound(children=[s,
                              Pos(0, cad.w * 2.5, 0) * b,
                              Pos(cad.L / 2, -cad.w * 2.0, 0) * sl])


if __name__ == "__main__":
    cad = Cad(YZipperParams())
    outdir = Path(__file__).resolve().parent / "out"
    outdir.mkdir(exist_ok=True)
    for name, part in build_all(cad).items():
        solids = part.solids()
        print(f"{name:15s} solids={len(solids):3d}  bbox={part.bounding_box().size}")
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
    export_step(assembly(cad), str(outdir / "assembly.step"))
    print(f"wrote {outdir}/*.step, *.stl, assembly.step")
