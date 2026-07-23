"""
Parametric CAD for the ZIP-CHAIN LINEAR ACTUATOR modelled in zipchain.py.

The mechanism: two chain strands feed through a guide head where a sprocket presses
them together into one rigid column that pushes and pulls; reverse the sprocket and
they separate and re-coil.  Each strand is modelled as a watertight COMB — an outer
side-spine plate carrying inward FINGERS at alternating pitch slots.  Bring the two
combs together (mirrored, half-pitch staggered) and the fingers INTERLACE across the
column, tiling it with print clearance — that's the "zip".

Parts (each a single watertight solid; key dims shared with ZipChainParams so CAD ↔
statics agree):
  strand    one comb: an outer spine plate (full length) + fingers protruding inward
            every other pitch slot.  Print two (the second rotated 180° about Z).
  column    the two strands interlaced into the deployed rigid column (assembly view)
  sprocket  the zip sprocket: a bored disc with z trapezoidal teeth at pitch radius
  head      guide/merge housing: a block with a rectangular column channel + sprocket
            axle bore + mount holes — the part that constrains the weak (buckling) axis

Frame: Z = column / deploy axis (up), X = seam direction (the two spines sit at ±X),
Y = sprocket axle.  mm throughout.

Run:  ../.venv/bin/python zipchain/cad.py
Out:  zipchain/out/*.step, *.stl, assembly.step
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from build123d import (
    BuildPart, Box, Cylinder, Locations, Pos, Rot, Compound, Mode, Align,
    export_step, export_stl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zipchain import ZipChainParams

_CTR = (Align.CENTER, Align.CENTER, Align.CENTER)
_MINZ = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Cad:
    P: ZipChainParams

    # column sample
    n_links: int = 16                # links per strand in the CAD sample
    seam_clr: float = 0.4            # X clearance between a finger tip and the far spine
    z_clr: float = 0.6              # Z clearance between interlacing fingers
    spine_t: float = 3.0             # outer side-spine plate thickness (X) [mm]
    finger_h_frac: float = 0.9       # finger height as fraction of column height (Y)

    # sprocket
    sprocket_t: float = 12.0         # sprocket thickness (Y) [mm]
    tooth_h: float = 4.0             # radial tooth height [mm]
    bore_dia: float = 6.0            # axle bore [mm]

    # head / guide
    head_len: float = 26.0           # along Z [mm]
    head_wall: float = 5.0           # wall around the column channel [mm]
    chan_clr: float = 0.6            # clearance around the column in the channel [mm]
    mount_hole: float = 3.2          # M3 clearance [mm]

    @property
    def col_w(self) -> float:
        return self.P.col_w_mm

    @property
    def col_h(self) -> float:
        return self.P.col_h_mm

    @property
    def pitch(self) -> float:
        return self.P.pitch_mm

    @property
    def total_len(self) -> float:
        return self.n_links * self.pitch


def make_strand(cad: Cad, mirror: bool = False) -> BuildPart:
    """One comb strand: outer spine plate + interlacing fingers on alternating slots.

    mirror=False -> spine on -X, fingers reach +X, fingers on EVEN z-slots.
    mirror=True  -> spine on +X, fingers reach -X, fingers on ODD  z-slots (staggered).
    """
    w2 = cad.col_w / 2.0
    h = cad.col_h
    p = cad.pitch
    L = cad.total_len
    fh = cad.finger_h_frac * h
    finger_len = p - cad.z_clr                      # along Z, leaves z-clearance
    finger_reach = cad.col_w - cad.seam_clr         # X span from own spine toward far spine
    sign = -1.0 if not mirror else 1.0              # spine side
    slot0 = 0 if not mirror else 1                  # even vs odd slots -> half-pitch stagger

    with BuildPart() as part:
        # outer side spine plate, full length along Z, on the ±X outer edge
        spine_x = sign * (w2 + cad.spine_t / 2.0)
        with Locations(Pos(spine_x, 0, L / 2.0)):
            Box(cad.spine_t, h, L, align=_CTR)
        # fingers reaching inward across the column, every other slot
        for k in range(slot0, cad.n_links, 2):
            zc = (k + 0.5) * p
            # finger inner face starts at own spine inner edge (±w2) and reaches across
            # centre at own_spine_edge -/+ reach/2
            fx = sign * (w2 - finger_reach / 2.0)
            with Locations(Pos(fx, 0, zc)):
                Box(finger_reach, fh, finger_len, align=_CTR)
    return part


def make_sprocket(cad: Cad) -> BuildPart:
    """Bored disc with z trapezoidal teeth at the pitch radius (spins about Y)."""
    P = cad.P
    r_p = P.r_pitch_m * 1000.0
    z = P.sprocket_teeth
    t = cad.sprocket_t
    r_root = r_p - cad.tooth_h / 2.0
    with BuildPart() as part:
        # hub disc — axis along Y
        with Locations(Rot(90, 0, 0)):
            Cylinder(radius=r_root, height=t, align=_CTR)
        # teeth around the rim
        import math
        for i in range(z):
            ang = 360.0 * i / z
            rx = (r_p) * math.cos(math.radians(ang))
            rz = (r_p) * math.sin(math.radians(ang))
            with Locations(Pos(rx, 0, rz) * Rot(0, ang, 0) * Rot(90, 0, 0)):
                # small radial tooth block
                Box(cad.pitch * 0.5, t, cad.tooth_h, align=_CTR, mode=Mode.ADD)
        # axle bore along Y
        with Locations(Rot(90, 0, 0)):
            Cylinder(radius=cad.bore_dia / 2.0, height=t + 2, align=_CTR, mode=Mode.SUBTRACT)
    return part


def make_head(cad: Cad) -> BuildPart:
    """Guide/merge housing: block + rectangular column channel + sprocket axle bore."""
    P = cad.P
    cw = cad.col_w + cad.chan_clr
    ch = cad.col_h + cad.chan_clr
    bw = cw + 2 * cad.head_wall
    bh = ch + 2 * cad.head_wall
    L = cad.head_len
    r_p = P.r_pitch_m * 1000.0
    with BuildPart() as part:
        with Locations(Pos(0, 0, L / 2.0)):
            Box(bw, bh, L, align=_CTR)
        # column channel straight through (Z)
        with Locations(Pos(0, 0, L / 2.0)):
            Box(cw, ch, L + 2, align=_CTR, mode=Mode.SUBTRACT)
        # sprocket axle bore (Y), positioned so a sprocket rim reaches the seam
        with Locations(Pos(0, 0, L / 2.0) * Rot(90, 0, 0)):
            Cylinder(radius=cad.bore_dia / 2.0 + 0.2, height=bh + 2, align=_CTR, mode=Mode.SUBTRACT)
        # 4 mount holes through the base wall (Z-through corners)
        ox = bw / 2.0 - cad.head_wall / 2.0
        oy = bh / 2.0 - cad.head_wall / 2.0
        for sx in (-ox, ox):
            for sy in (-oy, oy):
                with Locations(Pos(sx, sy, L / 2.0)):
                    Box(cad.mount_hole, cad.mount_hole, L + 2, align=_CTR, mode=Mode.SUBTRACT)
    return part


def build_all(cad: Cad) -> dict:
    return {
        "strand": make_strand(cad, mirror=False),
        "sprocket": make_sprocket(cad),
        "head": make_head(cad),
    }


def assembly(cad: Cad) -> Compound:
    a = make_strand(cad, mirror=False).part
    b = make_strand(cad, mirror=True).part
    head = make_head(cad).part
    spr = make_sprocket(cad).part
    # deploy the column upward out of the head sitting at the base
    z_head = 0.0
    col_shift = Pos(0, 0, cad.head_len + 4)
    parts = [
        head.moved(Pos(0, 0, z_head)),
        a.moved(col_shift),
        b.moved(col_shift),
        spr.moved(Pos(0, 0, cad.head_len / 2.0)),
    ]
    return Compound(children=parts)


if __name__ == "__main__":
    cad = Cad(ZipChainParams())
    outdir = Path(__file__).resolve().parent / "out"
    outdir.mkdir(exist_ok=True)

    parts = build_all(cad)
    for name, bp in parts.items():
        solids = bp.part.solids()
        print(f"{name:10s} solids={len(solids):2d}  bbox={bp.part.bounding_box().size}")
        export_step(bp.part, str(outdir / f"{name}.step"))
        export_stl(bp.part, str(outdir / f"{name}.stl"))

    asm = assembly(cad)
    export_step(asm, str(outdir / "assembly.step"))
    print(f"wrote {outdir}/*.step, *.stl, assembly.step")
