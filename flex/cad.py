"""
Parametric CAD of the 2-DOF tendon-driven gimbal modelled in flex.py — a pan/tilt platform on
a cross-trunnion gimbal, pulled by 3 tendons at 120° off 3 capstan spools, recentred by a spring.

Parts (each a single watertight solid; dimensions shared with FlexParams so CAD ↔ statics agree):
  base          disc + central pivot boss + 3 capstan-motor mounts at 120°
  pivot_post    central column up to the gimbal, topped by a fork (the X hinge)
  gimbal_ring   ring with 4 integral trunnions — X pair into the post fork, Y pair into the platform
  platform      disc with the Y-hinge prongs under it and 3 tendon ears at r_p on top (+ tool boss)
  capstan       flanged spool the tendon winds on  (×3)
  motor         representative gearmotor body       (×3)
  spring        a compression coil around the pivot post (recentres the platform)
  tendon        thin strand from ear to spool        (×3, cosmetic)

Frame: Z up, gimbal pivot at (0,0,h). sim.py animates platform tilt + capstan spin.

Run:  ../.venv/bin/python flex/cad.py
Out:  flex/out/*.step, *.stl, assembly.step
"""

import sys
from dataclasses import dataclass
from math import pi, cos, sin
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, Cylinder, Box, Hole, Helix, sweep, Circle, Plane,
    Locations, PolarLocations, Pos, Compound, Mode, Align, export_step, export_stl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flex import FlexParams

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Cad:
    p: FlexParams = None
    base_dia: float = 46.0
    base_t: float = 5.0
    post_dia: float = 8.0
    ring_dia: float = 18.0
    ring_t: float = 4.0            # ring radial thickness / axial height
    trunnion_dia: float = 2.6
    trunnion_len: float = 3.0
    fork_gap: float = 20.0         # inner span of the post fork (holds the ring)
    fork_t: float = 3.0
    platform_dia: float = 30.0
    platform_t: float = 4.0
    ear_dia: float = 5.0           # tendon-ear diameter
    ear_hole: float = 1.2
    tool_boss_dia: float = 6.0
    tool_boss_h: float = 6.0
    capstan_flange: float = 12.0
    capstan_flange_t: float = 1.2
    motor_dia: float = 12.0
    motor_len: float = 22.0
    tendon_dia: float = 0.8

    def __post_init__(self):
        if self.p is None:
            self.p = FlexParams()

    # ---- key heights (mm) -----------------------------------------------------
    @property
    def h(self): return self.p.pivot_h_mm          # gimbal pivot height
    @property
    def r_b(self): return self.p.base_r_mm
    @property
    def r_p(self): return self.p.platform_r_mm
    @property
    def r_cap(self): return self.p.capstan_r_mm

    def phis(self):
        return [i * 2 * pi / self.p.n_tendons for i in range(self.p.n_tendons)]


# --------------------------------------------------------------------------- #
# PARTS
# --------------------------------------------------------------------------- #
def make_base(c: Cad):
    with BuildPart() as b:
        Cylinder(radius=c.base_dia / 2, height=c.base_t, align=_MIN)
        # central boss the pivot post seats into
        with Locations((0, 0, c.base_t)):
            Cylinder(radius=c.post_dia / 2 + 2, height=3, align=_MIN)
        # 3 capstan-motor mount posts at the tendon base radius
        with PolarLocations(c.r_b, c.p.n_tendons):
            with Locations((0, 0, c.base_t)):
                Cylinder(radius=c.motor_dia / 2 + 1.5, height=5, align=_MIN)
        # motor bore down through each mount
        with PolarLocations(c.r_b, c.p.n_tendons):
            with Locations((0, 0, c.base_t + 5)):
                Cylinder(radius=c.motor_dia / 2, height=c.motor_len,
                         align=(Align.CENTER, Align.CENTER, Align.MAX), mode=Mode.SUBTRACT)
    return b.part


def make_pivot_post(c: Cad):
    fork_h = c.ring_dia + 4
    with BuildPart() as b:
        Cylinder(radius=c.post_dia / 2, height=c.h - fork_h / 2, align=_MIN)
        # fork: two prongs straddling the ring on the X axis, with trunnion holes
        top = c.h - fork_h / 2
        # yoke base ties the column to both prongs (else the prongs float free)
        with Locations((0, 0, top)):
            Box(c.fork_gap + 2 * c.fork_t, c.ring_dia + 2, 3, align=_MIN)
        for sgn in (+1, -1):
            with Locations((sgn * (c.fork_gap / 2 + c.fork_t / 2), 0, top)):
                Box(c.fork_t, c.ring_dia + 2, fork_h, align=_MIN)
        with Locations((0, 0, c.h)):
            Cylinder(radius=c.trunnion_dia / 2 + 0.15, height=c.fork_gap + 2 * c.fork_t + 2,
                     rotation=(0, 90, 0), mode=Mode.SUBTRACT)
    return b.part


def make_gimbal_ring(c: Cad):
    with BuildPart() as b:
        # a squared ring (torus-ish) — outer box minus inner window, centred at pivot
        Box(c.ring_dia, c.ring_dia, c.ring_t, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        Box(c.ring_dia - 2 * c.ring_t, c.ring_dia - 2 * c.ring_t, c.ring_t + 2,
            mode=Mode.SUBTRACT)
        # X-axis trunnions (into the post fork)
        with Locations((0, 0, 0)):
            Cylinder(radius=c.trunnion_dia / 2, height=c.ring_dia + 2 * c.trunnion_len,
                     rotation=(0, 90, 0))
        # Y-axis trunnions (into the platform prongs)
        with Locations((0, 0, 0)):
            Cylinder(radius=c.trunnion_dia / 2, height=c.ring_dia + 2 * c.trunnion_len,
                     rotation=(90, 0, 0))
    return b.part


def make_platform(c: Cad):
    prong_h = c.ring_dia + 4
    with BuildPart() as b:
        # main disc
        Cylinder(radius=c.platform_dia / 2, height=c.platform_t, align=_MIN)
        # tool boss on top centre
        with Locations((0, 0, c.platform_t)):
            Cylinder(radius=c.tool_boss_dia / 2, height=c.tool_boss_h, align=_MIN)
        # 3 tendon ears at r_p with a thru hole
        with PolarLocations(c.r_p, c.p.n_tendons):
            Cylinder(radius=c.ear_dia / 2, height=c.platform_t, align=_MIN)
            Hole(radius=c.ear_hole / 2)
        # two prongs hanging down on the Y axis to the gimbal ring
        for sgn in (+1, -1):
            with Locations((0, sgn * (c.fork_gap / 2 + c.fork_t / 2), -prong_h + c.platform_t)):
                Box(c.ring_dia + 2, c.fork_t, prong_h, align=_MIN)
        # Y trunnion holes in the prongs
        with Locations((0, 0, -prong_h + c.platform_t + prong_h / 2)):
            Cylinder(radius=c.trunnion_dia / 2 + 0.15, height=c.fork_gap + 2 * c.fork_t + 2,
                     rotation=(90, 0, 0), mode=Mode.SUBTRACT)
    return b.part


def make_capstan(c: Cad):
    with BuildPart() as b:
        Cylinder(radius=c.r_cap, height=10, align=_MIN)
        with Locations((0, 0, 0), (0, 0, 10 - c.capstan_flange_t)):
            Cylinder(radius=c.capstan_flange / 2, height=c.capstan_flange_t, align=_MIN)
        Cylinder(radius=1.5, height=12, align=(Align.CENTER, Align.CENTER, Align.CENTER),
                 mode=Mode.SUBTRACT)
    return b.part


def make_motor(c: Cad):
    with BuildPart() as b:
        Cylinder(radius=c.motor_dia / 2, height=c.motor_len, align=_MIN)
        with Locations((0, 0, c.motor_len)):
            Cylinder(radius=1.5, height=5, align=_MIN)     # shaft
    return b.part


def make_spring(c: Cad):
    """A compression coil around the pivot post (recentres the platform)."""
    turns = 6
    height = c.h - 6
    coil_r = c.post_dia / 2 + 3
    helix = Helix(pitch=height / turns, height=height, radius=coil_r)
    with BuildPart() as b:
        with BuildSketch(Plane(origin=helix @ 0, z_dir=helix % 0)):
            Circle(0.7)
        sweep(path=helix)
    return b.part


# --------------------------------------------------------------------------- #
# VALIDATION + ASSEMBLY
# --------------------------------------------------------------------------- #
def validate(c: Cad) -> bool:
    print("validating flex-gimbal geometry ...")
    ok = True
    checks = [
        ("pivot height clears the base + posts", c.h > c.base_t + 8, f"h {c.h} mm"),
        ("ring fits in the fork gap", c.ring_dia < c.fork_gap + 1e-6,
         f"ring {c.ring_dia} vs gap {c.fork_gap}"),
        ("tendon ears inside the platform disc", 2 * c.r_p + c.ear_dia < c.platform_dia,
         f"2·r_p+ear {2*c.r_p+c.ear_dia:.0f} < Ø{c.platform_dia}"),
        ("capstans + motors fit on the base", 2 * (c.r_b + c.motor_dia / 2 + 1.5) < c.base_dia,
         f"reach {2*(c.r_b+c.motor_dia/2+1.5):.0f} < Ø{c.base_dia}"),
        ("3 tendons for the 2-DOF joint", c.p.n_tendons >= 3, f"{c.p.n_tendons}"),
    ]
    for name, passed, detail in checks:
        print(f"  [{'ok ' if passed else 'XX '}] {name:<40} {detail}")
        ok = ok and passed
    print(f"  -> {'VALID' if ok else 'INVALID'}\n")
    return ok


def build(c: Cad, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "base": make_base(c),
        "pivot_post": make_pivot_post(c),
        "gimbal_ring": make_gimbal_ring(c),
        "platform": make_platform(c),
        "capstan": make_capstan(c),
        "motor": make_motor(c),
        "spring": make_spring(c),
    }
    for name, part in parts.items():
        n = len(part.solids())
        flag = "" if n == 1 else f"  <-- {n} solids!"
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name:<12} ({n} solid){flag}")

    h = c.h
    placed = [
        ("base", Pos(0, 0, 0) * parts["base"]),
        ("pivot_post", Pos(0, 0, 0) * parts["pivot_post"]),
        ("spring", Pos(0, 0, 3) * parts["spring"]),
        ("gimbal_ring", Pos(0, 0, h) * parts["gimbal_ring"]),
        ("platform", Pos(0, 0, h) * parts["platform"]),
    ]
    for i, phi in enumerate(c.phis()):
        x, y = c.r_b * cos(phi), c.r_b * sin(phi)
        placed.append((f"motor_{i}", Pos(x, y, c.base_t + 5 - c.motor_len) * parts["motor"]))
        placed.append((f"capstan_{i}", Pos(x, y, c.base_t + 6) * parts["capstan"]))
    bodies = []
    for label, body in placed:
        body.label = label
        bodies.append(body)
    export_step(Compound(children=bodies), str(outdir / "assembly.step"))
    print(f"  wrote assembly.step  ({len(bodies)} bodies)")


def report(c: Cad):
    print(f"=== 2-DOF tendon gimbal — Ø{c.base_dia:.0f} base, pivot {c.h:.0f} mm, "
          f"platform Ø{c.platform_dia:.0f} ===")
    print(f"  3 capstan motors (Ø{c.motor_dia:.0f}×{c.motor_len:.0f}) at r_b={c.r_b:.0f} mm; "
          f"tendon ears at r_p={c.r_p:.0f} mm")
    print(f"  cross-trunnion gimbal (Ø{c.trunnion_dia} pins), centering coil spring\n")


if __name__ == "__main__":
    c = Cad()
    report(c)
    if validate(c):
        out = Path(__file__).resolve().parent / "out"
        print(f"building -> {out}")
        build(c, out)
    else:
        sys.exit(1)
