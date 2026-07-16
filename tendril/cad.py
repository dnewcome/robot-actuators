"""
Parametric CAD for the SEGMENTED servo-driven tendril modelled in tendril.py — a printable test
rig: a flexible TPU finger (rigid vertebrae on a thin continuous spine) that BOLTS (2× M3) into a
rigid PLA mount, curled by one MG996R whose shaft runs ACROSS the finger axis so the horn spins in
the bending plane and the strings drop straight down the axis (no 90° cable bend).

Parts (each a single watertight solid; key dims shared with TendrilParams so CAD ↔ statics agree):
  tendril   TPU finger: bolt flange (2× M3) + a thin continuous SPINE carrying N rigid VERTEBRAE
            with gaps between them (bending happens only at the thin spine); two straight tendon
            channels at ±d run through the vertebrae, joined by a transverse tie-off hole at the tip
  mount     PLA bracket: foot + a vertical backplate that holds the MG996R with its shaft HORIZONTAL
            (tab screws + shaft/boss hole), and a top shelf cantilevered over the horn carrying 2
            string guides + the tendril's M3 pattern
  servo     representative MG996R body + boss (shaft along Y, static)
  horn      disc+arm horn with string holes at ±r_h, spins about Y in the bending plane (sim)

Frame: Z = finger axis (up), X = bending direction, Y = servo shaft. The horn spins about Y at
(0, y_horn, z_horn); the finger bolts to the shelf directly above it. mm throughout.

Run:  ../.venv/bin/python tendril/cad.py
Out:  tendril/out/*.step, *.stl, assembly.step
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from build123d import (
    BuildPart, Box, Cylinder, Locations, Pos, Compound, Mode, Align,
    export_step, export_stl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tendril import TendrilParams

_MINZ = (Align.CENTER, Align.CENTER, Align.MIN)
_MIN = (Align.MIN, Align.MIN, Align.MIN)
_CTR = (Align.CENTER, Align.CENTER, Align.CENTER)


@dataclass
class Cad:
    p: TendrilParams = field(default_factory=TendrilParams)

    # --- TPU finger flange / channels ---
    flange_l: float = 18.0     # X
    flange_w: float = 20.0     # Y
    flange_t: float = 4.0
    y_bolt: float = 8.0        # M3 flange bolts flank the vertebrae in Y
    m3_clear: float = 1.7
    m3_tap: float = 1.35
    chan_r: float = 0.8

    # --- MG996R servo (shaft along Y) ---
    servo_x: float = 40.7
    servo_z: float = 19.7
    servo_y: float = 37.0      # depth along the shaft
    shaft_off: float = 9.0     # output shaft offset from body center (X)
    tab_span: float = 49.5
    boss_dia: float = 10.0
    horn_disc_r: float = 12.0
    horn_t: float = 4.0

    # --- PLA mount ---
    plate_ty: float = 4.0      # backplate thickness (Y)
    z_horn: float = 15.0       # shaft height above the foot
    foot_t: float = 4.0
    shelf_t: float = 4.0
    shelf_l: float = 24.0      # X
    shelf_w: float = 22.0      # Y
    y_horn: float = 7.0        # horn / finger Y offset in front of the backplate
    guide_r: float = 1.2

    # ---- derived (mm) ---------------------------------------------------------
    @property
    def d_off(self): return self.p.d_off_mm
    @property
    def z_base(self): return self.z_horn + self.horn_disc_r + 6.0   # shelf underside (clears horn)
    @property
    def shelf_top(self): return self.z_base + self.shelf_t
    @property
    def tab_x_a(self): return -self.shaft_off + self.tab_span / 2
    @property
    def tab_x_b(self): return -self.shaft_off - self.tab_span / 2


# --------------------------------------------------------------------------- #
# PARTS
# --------------------------------------------------------------------------- #
def make_tendril(c: Cad):
    """TPU finger: flange + continuous spine carrying rigid vertebrae + two tendon channels."""
    p = c.p
    fb = c.flange_t
    L = p.length_mm
    pitch = p.seg_len_mm + p.gap_mm
    z_last = fb + (p.n_vert - 1) * pitch
    tie_z = z_last + p.seg_len_mm * 0.5
    with BuildPart() as b:
        Box(c.flange_l, c.flange_w, fb, align=_MINZ)                       # flange
        with Locations((0, 0, fb - 1)):                                   # continuous spine
            Box(p.spine_t_mm, p.spine_w_mm, L + 1, align=_MINZ)
        for i in range(p.n_vert):                                         # rigid vertebrae
            with Locations((0, 0, fb + i * pitch)):
                Box(p.seg_t_mm, p.seg_w_mm, p.seg_len_mm, align=_MINZ)
        for sx in (+1, -1):                                               # tendon channels
            with Locations((sx * c.d_off, 0, 0)):
                Cylinder(radius=c.chan_r, height=tie_z, align=_MINZ, mode=Mode.SUBTRACT)
        with Locations((0, 0, tie_z)):                                    # tip tie-off (loop crossover)
            Cylinder(radius=c.chan_r, height=2 * c.d_off + 4, rotation=(0, 90, 0),
                     align=_CTR, mode=Mode.SUBTRACT)
        for sy in (+1, -1):                                               # M3 flange holes
            with Locations((0, sy * c.y_bolt, 0)):
                Cylinder(radius=c.m3_clear, height=fb + 1, align=_MINZ, mode=Mode.SUBTRACT)
    return b.part


def make_mount(c: Cad):
    """PLA bracket: foot + vertical backplate (holds the horizontal servo) + tendril shelf."""
    with BuildPart() as b:
        # foot (stable base; reaches behind the servo and in front under the finger)
        with Locations((-40.0, -(c.plate_ty + c.servo_y + 4), 0)):
            Box(62.0, c.plate_ty + c.servo_y + 4 + c.y_horn + 14, c.foot_t, align=_MIN)
        # backplate (XZ), y in [-plate_ty, 0]
        with Locations((-40.0, -c.plate_ty, 0)):
            Box(62.0, c.plate_ty, c.z_base, align=_MIN)
        # shelf (XY) at z_base, over the horn
        with Locations((-c.shelf_l / 2, -4.0, c.z_base)):
            Box(c.shelf_l, c.shelf_w, c.shelf_t, align=_MIN)
        # shaft/boss clearance hole through the backplate (along Y)
        with Locations((0, -c.plate_ty / 2, c.z_horn)):
            Cylinder(radius=c.boss_dia / 2, height=c.plate_ty + 2, rotation=(90, 0, 0),
                     align=_CTR, mode=Mode.SUBTRACT)
        # servo mounting-tab screw holes
        for tx in (c.tab_x_a, c.tab_x_b):
            with Locations((tx, -c.plate_ty / 2, c.z_horn)):
                Cylinder(radius=c.m3_tap, height=c.plate_ty + 2, rotation=(90, 0, 0),
                         align=_CTR, mode=Mode.SUBTRACT)
        # string guide holes through the shelf (align to the channel entries)
        for sx in (+1, -1):
            with Locations((sx * c.d_off, c.y_horn, c.z_base + c.shelf_t / 2)):
                Cylinder(radius=c.guide_r, height=c.shelf_t + 2, align=_CTR, mode=Mode.SUBTRACT)
        # tendril M3 bolt holes through the shelf
        for sy in (+1, -1):
            with Locations((0, c.y_horn + sy * c.y_bolt, c.z_base + c.shelf_t / 2)):
                Cylinder(radius=c.m3_tap, height=c.shelf_t + 2, align=_CTR, mode=Mode.SUBTRACT)
    return b.part


def make_servo(c: Cad):
    """Representative MG996R with the shaft along +Y (body behind the backplate)."""
    with BuildPart() as b:
        with Locations((-c.shaft_off, -(c.plate_ty + c.servo_y / 2), c.z_horn)):
            Box(c.servo_x, c.servo_y, c.servo_z, align=_CTR)
        # output boss poking forward through the plate
        with Locations((0, -c.plate_ty / 2 + 2, c.z_horn)):
            Cylinder(radius=c.boss_dia / 2 - 0.6, height=c.plate_ty + 6, rotation=(90, 0, 0),
                     align=_CTR)
    return b.part


def make_horn(c: Cad):
    """Servo horn: disc + arm with string holes at ±r_h, spins about Y (sim)."""
    r_h = c.p.horn_r_mm
    with BuildPart() as b:
        Cylinder(radius=c.horn_disc_r, height=c.horn_t, rotation=(90, 0, 0), align=_CTR)
        Box(2 * (r_h + 3), c.horn_t, 6.0, align=_CTR)
        Cylinder(radius=2.5, height=c.horn_t + 2, rotation=(90, 0, 0), align=_CTR, mode=Mode.SUBTRACT)
        for sx in (+1, -1):
            with Locations((sx * r_h, 0, 0)):
                Cylinder(radius=0.8, height=c.horn_t + 2, rotation=(90, 0, 0),
                         align=_CTR, mode=Mode.SUBTRACT)
    return b.part


# --------------------------------------------------------------------------- #
# VALIDATION + ASSEMBLY
# --------------------------------------------------------------------------- #
def validate(c: Cad) -> bool:
    print("validating segmented tendril CAD geometry ...")
    ok = True
    p = c.p
    checks = [
        ("channels fit inside the vertebra thickness",
         c.d_off + c.chan_r + 0.4 < p.seg_t_mm / 2,
         f"d {c.d_off} + r {c.chan_r} + wall < t/2 {p.seg_t_mm/2:.1f}"),
        ("M3 flange bolts clear the vertebrae",
         c.y_bolt - c.m3_clear > p.seg_w_mm / 2,
         f"bolt @ {c.y_bolt} vs vertebra half-width {p.seg_w_mm/2:.1f}"),
        ("flange wider than a vertebra",
         c.flange_w > p.seg_w_mm and c.flange_l > p.seg_t_mm,
         f"flange {c.flange_l}×{c.flange_w} vs vertebra {p.seg_t_mm}×{p.seg_w_mm}"),
        ("shelf clears the horn (height)",
         c.z_base > c.z_horn + c.horn_disc_r,
         f"z_base {c.z_base:.0f} > horn top {c.z_horn + c.horn_disc_r:.0f}"),
        ("horn (in bending plane) clears the backplate",
         c.y_horn - c.horn_t / 2 > 0,
         f"horn front face y {c.y_horn - c.horn_t/2:.0f} > 0"),
        ("shelf spans the tendril flange + bolts",
         c.shelf_w - 4 > c.y_horn + c.y_bolt and c.shelf_l > c.flange_l,
         f"shelf {c.shelf_l}×{c.shelf_w} vs flange @ y_horn {c.y_horn}"),
    ]
    for name, passed, detail in checks:
        print(f"  [{'ok ' if passed else 'XX '}] {name:<42} {detail}")
        ok = ok and passed
    print(f"  -> {'VALID' if ok else 'INVALID'}\n")
    return ok


def build(c: Cad, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "tendril": make_tendril(c),
        "mount": make_mount(c),
        "servo": make_servo(c),
        "horn": make_horn(c),
    }
    for name, part in parts.items():
        n = len(part.solids())
        flag = "" if n == 1 else f"  <-- {n} solids!"
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name:<10} ({n} solid){flag}")

    placed = [
        ("mount", Pos(0, 0, 0) * parts["mount"]),
        ("servo", Pos(0, 0, 0) * parts["servo"]),
        ("horn", Pos(0, c.y_horn, c.z_horn) * parts["horn"]),
        ("tendril", Pos(0, c.y_horn, c.shelf_top) * parts["tendril"]),
    ]
    bodies = []
    for label, body in placed:
        body.label = label
        bodies.append(body)
    export_step(Compound(children=bodies), str(outdir / "assembly.step"))
    print(f"  wrote assembly.step  ({len(bodies)} bodies)")


def report(c: Cad):
    p = c.p
    print(f"=== segmented tendril CAD — MG996R (shaft horizontal) + TPU finger L{p.length_mm:.0f} ===")
    print(f"  TPU finger: flange {c.flange_l}×{c.flange_w}×{c.flange_t} (2× M3 @ ±{c.y_bolt}), "
          f"{p.n_vert} vertebrae on a {p.spine_t_mm}×{p.spine_w_mm} spine, 2 channels ±{c.d_off}")
    print(f"  PLA mount: foot + backplate (servo shaft along Y @ z={c.z_horn:.0f}) + shelf @ "
          f"z={c.z_base:.0f}; horn r_h={p.horn_r_mm}, guides ±{c.d_off} mm — strings run straight up\n")


if __name__ == "__main__":
    c = Cad()
    report(c)
    if validate(c):
        out = Path(__file__).resolve().parent / "out"
        print(f"building -> {out}")
        build(c, out)
    else:
        sys.exit(1)
