"""
Parametric CAD for the servo-driven continuum tendril modelled in tendril.py — a printable test
rig: a flexible TPU finger that BOLTS (M3) into a rigid PLA mount, pulled by one MG996R hobby
servo whose horn reels an antagonistic string loop.

Parts (each a single watertight solid; key dims shared with TendrilParams so CAD ↔ statics agree):
  tendril   flexible TPU finger: bolt flange (2× M3) + tapered lofted body + two tapered tendon
            channels (±d, narrowing with the taper) joined by a transverse tie-off hole at the tip
            → one Dyneema loop runs horn → channel A → tip → channel B → horn (the antagonist pair)
  mount     PLA bracket: base plate with a MG996R body pocket + tab screws, two posts up to a top
            deck; the deck carries 2 string guide holes and the tendril's M3 bolt pattern
  servo     representative MG996R body + output boss (static, for fit/sim; not printed)
  horn      the servo horn — a short bar with holes at r_h (spins in sim; not printed)

Frame: Z up. Servo shaft on Z at x=0; horn spins just under the deck; tendril bolts to the deck
top and points up, curling in the ±X (thickness) plane. mm throughout.

Run:  ../.venv/bin/python tendril/cad.py
Out:  tendril/out/*.step, *.stl, assembly.step
"""

import sys
from dataclasses import dataclass, field
from math import atan2, degrees
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, Box, Cylinder, Circle, Rectangle, loft,
    Locations, Pos, Compound, Mode, Align, Plane, export_step, export_stl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tendril import TendrilParams

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)
_CTR = (Align.CENTER, Align.CENTER, Align.CENTER)


@dataclass
class Cad:
    p: TendrilParams = field(default_factory=TendrilParams)

    # --- TPU finger flange / channels ---
    flange_l: float = 16.0     # X (thickness/bending direction)
    flange_w: float = 24.0     # Y (width direction)
    flange_t: float = 4.0
    y_bolt: float = 9.0        # M3 flange bolts flank the beam in Y
    m3_clear: float = 1.7      # Ø3.4 clearance
    m3_tap: float = 1.35       # Ø2.7 for a self-tap / heat-set pilot
    chan_r: float = 0.8        # tendon channel radius (Ø1.6 string clearance)

    # --- MG996R servo (representative) ---
    servo_l: float = 40.5      # X (body length)
    servo_w: float = 20.0      # Y (body width)
    servo_h: float = 37.0      # body height below the mounting tabs
    tab_span: float = 49.5     # mounting-hole center span (X)
    shaft_off: float = 9.0     # output shaft offset from body center toward +X
    boss_h: float = 6.0        # output boss height above the tab plane
    horn_dia: float = 12.0
    horn_t: float = 4.0

    # --- PLA mount ---
    plate_l: float = 58.0
    plate_w: float = 30.0
    plate_t: float = 3.0
    post_y: float = 12.0       # posts sit at ±post_y (clear of the horn)
    post_sz: float = 6.0
    deck_z0: float = 14.0      # deck underside height above the tab plane (clears the horn)
    deck_l: float = 24.0
    deck_w: float = 32.0
    deck_t: float = 4.0
    guide_r: float = 1.0       # string guide holes through the deck

    # ---- derived offsets (mm), shared with the physics ------------------------
    @property
    def d_base(self): return self.p.off_frac * self.p.t_base_mm / 2
    @property
    def t_sterm(self):
        """Beam thickness where the tendon terminates (channels end)."""
        return self.p.t_base_mm + (self.p.t_tip_mm - self.p.t_base_mm) * self.p.tendon_frac
    @property
    def d_sterm(self): return self.p.off_frac * self.t_sterm / 2
    @property
    def chan_top(self):
        """Height (from the flange bottom) where the tendon channels end = flange + driven length."""
        return self.flange_t + self.p.tendon_frac * self.p.length_mm
    @property
    def r_h(self): return self.p.horn_r_mm
    @property
    def deck_top(self): return self.deck_z0 + self.deck_t


# --------------------------------------------------------------------------- #
# PARTS
# --------------------------------------------------------------------------- #
def make_tendril(c: Cad):
    """Flexible TPU finger — flange + tapered lofted beam + two tapered tendon channels."""
    L = c.p.length_mm
    fb = c.flange_t
    top = fb + L
    with BuildPart() as b:
        # bolt flange
        Box(c.flange_l, c.flange_w, fb, align=_MIN)
        # tapered body: loft base rect -> tip rect (X = thickness, Y = width)
        with BuildSketch(Plane.XY.offset(fb)):
            Rectangle(c.p.t_base_mm, c.p.w_base_mm)
        with BuildSketch(Plane.XY.offset(top)):
            Rectangle(c.p.t_tip_mm, c.p.w_tip_mm)
        loft()
        # two tendon channels at ±d(s), tapering inward with the beam; they END at the tendon
        # termination (chan_top), leaving a solid passive tail above
        chan_top = c.chan_top
        for sx in (+1, -1):
            with BuildSketch(Plane.XY.offset(0.0)):
                with Locations((sx * c.d_base, 0)):
                    Circle(c.chan_r)
            with BuildSketch(Plane.XY.offset(chan_top)):
                with Locations((sx * c.d_sterm, 0)):
                    Circle(c.chan_r)
            loft(mode=Mode.SUBTRACT)
        # transverse tie-off hole joining the two channels at the termination (loop passes through)
        with Locations((0, 0, chan_top)):
            Cylinder(radius=c.chan_r, height=2 * c.d_base + 4, rotation=(0, 90, 0),
                     align=_CTR, mode=Mode.SUBTRACT)
        # 2 M3 flange bolt holes flanking the beam
        for sy in (+1, -1):
            with Locations((0, sy * c.y_bolt, 0)):
                Cylinder(radius=c.m3_clear, height=fb + 1, align=_MIN, mode=Mode.SUBTRACT)
    return b.part


def make_mount(c: Cad):
    """PLA bracket: servo base plate + posts + tendril deck."""
    with BuildPart() as b:
        # base plate (tab plane at z=0), centered on the servo pocket
        with Locations((-c.shaft_off, 0, -c.plate_t)):
            Box(c.plate_l, c.plate_w, c.plate_t, align=_MIN)
        # servo body through-pocket
        with Locations((-c.shaft_off, 0, -c.plate_t - 1)):
            Box(c.servo_l + 0.6, c.servo_w + 0.6, c.plate_t + 3, align=_MIN, mode=Mode.SUBTRACT)
        # servo mounting-tab screw holes
        for sx in (+1, -1):
            with Locations((-c.shaft_off + sx * c.tab_span / 2, 0, -c.plate_t - 1)):
                Cylinder(radius=c.m3_tap, height=c.plate_t + 3, align=_MIN, mode=Mode.SUBTRACT)
        # two posts up to the deck
        for sy in (+1, -1):
            with Locations((0, sy * c.post_y, 0)):
                Box(c.post_sz, c.post_sz, c.deck_z0, align=_MIN)
        # top deck
        with Locations((0, 0, c.deck_z0)):
            Box(c.deck_l, c.deck_w, c.deck_t, align=_MIN)
        # string guide holes through the deck (align to the channel bases)
        for sx in (+1, -1):
            with Locations((sx * c.d_base, 0, c.deck_z0 - 1)):
                Cylinder(radius=c.guide_r, height=c.deck_t + 2, align=_MIN, mode=Mode.SUBTRACT)
        # tendril M3 bolt holes through the deck (self-tap / heat-set)
        for sy in (+1, -1):
            with Locations((0, sy * c.y_bolt, c.deck_z0 - 1)):
                Cylinder(radius=c.m3_tap, height=c.deck_t + 2, align=_MIN, mode=Mode.SUBTRACT)
    return b.part


def make_servo(c: Cad):
    """Representative MG996R: body + mounting tabs + output boss (static)."""
    with BuildPart() as b:
        # body hangs below the tab plane (z=0)
        with Locations((-c.shaft_off, 0, -c.servo_h)):
            Box(c.servo_l, c.servo_w, c.servo_h, align=_MIN)
        # mounting tabs (thin, at the top of the body)
        with Locations((-c.shaft_off, 0, -3.0)):
            Box(c.tab_span + 7, c.servo_w, 2.5, align=_MIN)
        # output boss on Z at x=0
        Cylinder(radius=6.0, height=c.boss_h, align=_MIN)
    return b.part


def make_horn(c: Cad):
    """The servo horn — a short bar with attach holes at ±r_h (spins in sim)."""
    with BuildPart() as b:
        Box(2 * c.horn_dia, 6.0, c.horn_t, align=_MIN)
        Cylinder(radius=c.horn_dia / 2, height=c.horn_t, align=_MIN)
        # hub hole + two string holes at ±r_h
        Cylinder(radius=2.5, height=c.horn_t + 1, align=_MIN, mode=Mode.SUBTRACT)
        for sx in (+1, -1):
            with Locations((sx * c.r_h, 0, 0)):
                Cylinder(radius=0.8, height=c.horn_t + 1, align=_MIN, mode=Mode.SUBTRACT)
    return b.part


# --------------------------------------------------------------------------- #
# VALIDATION + ASSEMBLY
# --------------------------------------------------------------------------- #
def validate(c: Cad) -> bool:
    print("validating tendril CAD geometry ...")
    ok = True
    checks = [
        ("channels fit inside the thickness at the termination",
         c.d_sterm + c.chan_r + 0.4 < c.t_sterm / 2 + 1e-6,
         f"d {c.d_sterm:.1f} + r {c.chan_r} + wall < t/2 {c.t_sterm/2:.1f}"),
        ("flange wider than the beam base",
         c.flange_w > c.p.w_base_mm and c.flange_l > c.p.t_base_mm,
         f"flange {c.flange_l}×{c.flange_w} vs beam {c.p.t_base_mm}×{c.p.w_base_mm}"),
        ("M3 bolts clear the beam base",
         c.y_bolt - c.m3_clear > c.p.w_base_mm / 2,
         f"bolt @ {c.y_bolt} vs beam half-width {c.p.w_base_mm/2:.1f}"),
        ("posts clear the horn swing",
         c.post_y - c.post_sz / 2 > c.horn_dia / 2,
         f"post inner {c.post_y - c.post_sz/2:.0f} > horn r {c.horn_dia/2:.0f}"),
        ("deck clears the horn (height)",
         c.deck_z0 > c.boss_h + c.horn_t,
         f"deck z0 {c.deck_z0} > boss+horn {c.boss_h + c.horn_t:.0f}"),
        ("base plate spans the servo tabs",
         c.plate_l > c.tab_span + 4, f"plate {c.plate_l} > tab span {c.tab_span}+4"),
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
        ("servo", Pos(0, 0, 0) * parts["servo"]),
        ("horn", Pos(0, 0, c.boss_h) * parts["horn"]),
        ("mount", Pos(0, 0, 0) * parts["mount"]),
        ("tendril", Pos(0, 0, c.deck_top) * parts["tendril"]),
    ]
    bodies = []
    for label, body in placed:
        body.label = label
        bodies.append(body)
    export_step(Compound(children=bodies), str(outdir / "assembly.step"))
    print(f"  wrote assembly.step  ({len(bodies)} bodies)")


def report(c: Cad):
    print(f"=== servo-driven tendril CAD — MG996R + TPU finger L{c.p.length_mm:.0f} ===")
    print(f"  TPU finger: flange {c.flange_l}×{c.flange_w}×{c.flange_t} (2× M3 @ ±{c.y_bolt}), "
          f"tapered body, 2 channels ±{c.d_base:.1f}→{c.d_sterm:.1f} mm to {c.p.tendon_frac:.0%}L")
    print(f"  PLA mount: {c.plate_l}×{c.plate_w} base plate + posts + deck at z={c.deck_top:.0f}; "
          f"horn r_h={c.r_h} mm, guide holes ±{c.d_base:.1f} mm\n")


if __name__ == "__main__":
    c = Cad()
    report(c)
    if validate(c):
        out = Path(__file__).resolve().parent / "out"
        print(f"building -> {out}")
        build(c, out)
    else:
        sys.exit(1)
