"""
Parametric CAD of the LINEAR RAIL SERVO — a belt axis driven by a crappy TT gearmotor,
with the capacitive vernier scale (encoder.py) built right into the rail.

The whole conceit of this folder made physical: cheap round guide rods + a plastic TT
gearmotor + a GT2 belt give a sloppy, backlashy actuator; a two-PCB capacitive scale
along the rail (the digital-caliper trick) gives the ~1 µm feedback that closes the loop
(servo.py) and cancels the slop. So the interesting geometry is the SCALE PAIR:

  * scale PCB  — sits flat on the base along the whole travel, facing up. Two rows of
                 coupling electrodes at slightly different pitches P1, P2 (the vernier /
                 nonius pair) are engraved into its face — the beat of the two gives
                 absolute position over the travel.
  * slider PCB — hangs under the carriage, facing down, separated from the scale by a
                 thin air gap g. Carries the transmitter fingers + receiver comb.

Frame: X = travel (long axis), Y = transverse, Z = up, z=0 at the base underside.
Belt runs along X in a horizontal plane at rod height; the two GT2 pulleys are vertical
(Z axis) at the ends — the drive pulley on the (vertically-mounted) TT motor shaft.

Parts (each a single watertight solid; sim.py animates carriage + pulleys + belt):
  base, rod ×2, motor_block, idler_block, carriage, drive_pulley, idler_pulley,
  belt, scale_pcb (vernier pattern), slider_pcb, motor (TT, representative body).

Run:  ../.venv/bin/python linear-rail-servo/rail.py
Out:  linear-rail-servo/out/*.step, *.stl, assembly.step
"""

import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path

from build123d import (
    BuildPart, Box, Cylinder, Locations, Pos, Compound,
    Mode, Align, export_step, export_stl,
)

from encoder import CapVernierEncoder

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)
_TOP = (Align.CENTER, Align.CENTER, Align.MAX)


@dataclass
class Params:
    # --- travel & carriage -----------------------------------------------------
    travel: float = 150.0
    carriage_len: float = 46.0        # X
    carriage_w: float = 60.0          # Y
    carriage_h: float = 22.0          # Z
    end_clear: float = 8.0            # gap between carriage extreme and end block

    # --- guide rods ------------------------------------------------------------
    rod_dia: float = 8.0
    rod_span: float = 44.0            # Y centre-to-centre
    rod_z: float = 24.0               # rod axis height (also belt plane)
    bushing_od: float = 15.0          # LM8UU pocket OD (carriage bores)

    # --- base & end blocks -----------------------------------------------------
    base_t: float = 6.0
    base_w: float = 72.0
    block_x: float = 22.0             # end-block thickness (X)
    block_h: float = 32.0             # end-block height (holds rods + pulley)

    # --- belt / pulleys (GT2, vertical Z axis) ---------------------------------
    pulley_dia: float = 12.0          # pitch-ish body dia
    pulley_h: float = 10.0
    flange_dia: float = 16.0
    flange_t: float = 1.2
    belt_t: float = 1.4               # belt thickness (radial build)
    belt_h: float = 7.0
    motor_shaft_dia: float = 5.4      # TT double-D nominal

    # --- capacitive scale pair (the encoder) -----------------------------------
    scale_w: float = 24.0             # Y width of scale PCB (two tracks)
    scale_t: float = 1.6             # PCB thickness
    pcb_pad_depth: float = 0.3        # engraved electrode depth
    pad_y: float = 8.0                # electrode length (Y)
    track_offset: float = 6.0         # ±Y of the two vernier rows
    air_gap: float = 0.25             # scale face -> slider face
    slider_len: float = 34.0          # X of slider PCB
    slider_w: float = 20.0
    slider_t: float = 1.6

    enc: CapVernierEncoder = None

    def __post_init__(self):
        if self.enc is None:
            self.enc = CapVernierEncoder(travel_mm=self.travel)

    # ---- derived layout -------------------------------------------------------
    @property
    def rail_len(self) -> float:
        return self.travel + self.carriage_len + 2 * self.block_x + 2 * self.end_clear

    @property
    def x_motor(self) -> float:                 # motor end-block near face (x of its centre)
        return self.block_x / 2

    @property
    def x_idler(self) -> float:
        return self.rail_len - self.block_x / 2

    @property
    def pulley_cx_motor(self) -> float:
        return self.block_x + 6.0

    @property
    def pulley_cx_idler(self) -> float:
        return self.rail_len - self.block_x - 6.0

    @property
    def carriage_x0(self) -> float:             # carriage centre at travel = 0
        return self.block_x + self.end_clear + self.carriage_len / 2

    @property
    def scale_z(self) -> float:
        return self.base_t

    @property
    def scale_top(self) -> float:
        return self.base_t + self.scale_t

    @property
    def slider_bottom(self) -> float:
        return self.scale_top + self.air_gap

    @property
    def scale_len(self) -> float:
        return self.travel + self.slider_len     # slider stays over the scale everywhere

    @property
    def scale_x0(self) -> float:
        return self.pulley_cx_motor + self.pulley_dia   # start clear of the motor pulley


# --------------------------------------------------------------------------- #
# PARTS
# --------------------------------------------------------------------------- #

def make_base(p: Params):
    with BuildPart() as b:
        Box(p.rail_len, p.base_w, p.base_t, align=(Align.MIN, Align.CENTER, Align.MIN))
    return b.part


def make_end_block(p: Params):
    """One end block: a wall spanning the rod height with two rod bores + a pulley pilot."""
    with BuildPart() as b:
        Box(p.block_x, p.base_w, p.block_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # rod bores (through, along X)
        with Locations((0, +p.rod_span / 2, p.rod_z), (0, -p.rod_span / 2, p.rod_z)):
            Box(p.block_x + 2, p.rod_dia, p.rod_dia, mode=Mode.SUBTRACT)
        for y, z in [(+p.rod_span / 2, p.rod_z), (-p.rod_span / 2, p.rod_z)]:
            with Locations((0, y, z)):
                Cylinder(radius=p.rod_dia / 2, height=p.block_x + 2,
                         rotation=(0, 90, 0), mode=Mode.SUBTRACT)
    return b.part


def make_rod(p: Params):
    length = p.rail_len - p.block_x        # spans between the two end blocks (into bores)
    with BuildPart() as b:
        Cylinder(radius=p.rod_dia / 2, height=length, rotation=(0, 90, 0),
                 align=(Align.CENTER, Align.CENTER, Align.CENTER))
    return b.part


def make_pulley(p: Params):
    """GT2-style pulley: body + two flanges + bore, Z axis, single solid."""
    with BuildPart() as b:
        Cylinder(radius=p.pulley_dia / 2, height=p.pulley_h, align=_MIN)
        with Locations((0, 0, 0), (0, 0, p.pulley_h - p.flange_t)):
            Cylinder(radius=p.flange_dia / 2, height=p.flange_t, align=_MIN)
        Cylinder(radius=p.motor_shaft_dia / 2, height=p.pulley_h + 2,
                 align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)
    return b.part


def make_belt(p: Params):
    """A flat GT2 loop hugging both pulleys: racetrack outer minus racetrack inner."""
    c1, c2 = p.pulley_cx_motor, p.pulley_cx_idler
    r_in = p.pulley_dia / 2
    r_out = r_in + p.belt_t
    span = c2 - c1
    with BuildPart() as b:
        with Locations(((c1 + c2) / 2, 0, 0)):
            Box(span, 2 * r_out, p.belt_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((c1, 0, 0), (c2, 0, 0)):
            Cylinder(radius=r_out, height=p.belt_h, align=_MIN)
        # subtract inner racetrack
        with Locations(((c1 + c2) / 2, 0, -0.1)):
            Box(span, 2 * r_in, p.belt_h + 0.2,
                align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
        with Locations((c1, 0, -0.1), (c2, 0, -0.1)):
            Cylinder(radius=r_in, height=p.belt_h + 0.2, align=_MIN, mode=Mode.SUBTRACT)
    return b.part


def make_carriage(p: Params):
    """Rides the two rods; clamps the belt at centre; a tab hangs down to hold the slider."""
    z0 = p.rod_z - p.carriage_h / 2
    with BuildPart() as b:
        with Locations((0, 0, z0)):
            Box(p.carriage_len, p.carriage_w, p.carriage_h, align=_MIN)
        # rod bores (bushing pockets) through along X
        for y in (+p.rod_span / 2, -p.rod_span / 2):
            with Locations((0, y, p.rod_z)):
                Cylinder(radius=p.bushing_od / 2, height=p.carriage_len + 2,
                         rotation=(0, 90, 0), mode=Mode.SUBTRACT)
        # belt clamp slot at centre (a thin vertical slot the belt passes through)
        with Locations((0, 0, p.rod_z)):
            Box(p.carriage_len + 2, p.belt_t + 0.6, p.belt_h + 1.0, mode=Mode.SUBTRACT)
        # tab hanging down to the slider face
        tab_top = z0
        tab_bot = p.slider_bottom - p.slider_t     # slider mounts to tab underside
        tab_h = tab_top - tab_bot
        with Locations((0, 0, tab_bot)):
            Box(p.slider_len + 4, p.slider_w + 4, tab_h, align=_MIN)
    return b.part


def make_scale_pcb(p: Params):
    """The capacitive SCALE: a long PCB on the base, two rows of vernier electrodes
    (pitch P1 and P2) engraved into the top face. This is the encoder in the rail."""
    P1 = p.enc.pitch1_mm
    P2 = p.enc.pitch2_mm
    pad_x1 = P1 * 0.6
    pad_x2 = P2 * 0.6
    n1 = int(p.scale_len / P1)
    n2 = int(p.scale_len / P2)
    with BuildPart() as b:
        Box(p.scale_len, p.scale_w, p.scale_t, align=(Align.MIN, Align.CENTER, Align.MIN))
        # row A (pitch P1) at +track_offset ; engrave from the top face downward
        xs1 = [(i + 0.5) * P1 for i in range(n1) if (i + 0.5) * P1 < p.scale_len - 1]
        with Locations(*[(x, +p.track_offset, p.scale_t) for x in xs1]):
            Box(pad_x1, p.pad_y, p.pcb_pad_depth, align=_TOP, mode=Mode.SUBTRACT)
        # row B (pitch P2) at -track_offset
        xs2 = [(i + 0.5) * P2 for i in range(n2) if (i + 0.5) * P2 < p.scale_len - 1]
        with Locations(*[(x, -p.track_offset, p.scale_t) for x in xs2]):
            Box(pad_x2, p.pad_y, p.pcb_pad_depth, align=_TOP, mode=Mode.SUBTRACT)
    return b.part


def make_slider_pcb(p: Params):
    """Slider electrode PCB under the carriage, facing the scale across the air gap.
    A few transmitter fingers engraved for visual parity with the scale."""
    with BuildPart() as b:
        Box(p.slider_len, p.slider_w, p.slider_t, align=_MIN)
        # transmitter finger hints on the underside (face the scale)
        P1 = p.enc.pitch1_mm
        n = int(p.slider_len / (P1 / 4))
        xs = [(i + 0.5) * (P1 / 4) - p.slider_len / 2 for i in range(n)
              if 0.5 < (i + 0.5) * (P1 / 4) < p.slider_len - 0.5]
        with Locations(*[(x, 0, 0) for x in xs]):
            Box(P1 / 8, p.pad_y, p.pcb_pad_depth, align=_MIN, mode=Mode.SUBTRACT)
    return b.part


def make_motor(p: Params):
    """Representative TT gearmotor, mounted with the output shaft pointing UP (Z) so a
    Z-axis pulley drives the horizontal belt. Gearbox box + motor can + output shaft."""
    gb_l, gb_w, gb_h = 37.0, 22.6, 18.7
    with BuildPart() as b:
        # gearbox standing on its side: L along X, H(18.7) along Y, W(22.6) along Z
        Box(gb_l, gb_h, gb_w, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # motor can out the back (-X), horizontal axis along X
        with Locations((-gb_l / 2, 0, gb_w / 2)):
            Cylinder(radius=10.0, height=20.0, rotation=(0, -90, 0), align=_MIN)
        # output shaft up (+Z) from the top
        with Locations((gb_l / 2 - 8, 0, gb_w)):
            Cylinder(radius=p.motor_shaft_dia / 2, height=14.0, align=_MIN)
    return b.part


# --------------------------------------------------------------------------- #
# VALIDATION
# --------------------------------------------------------------------------- #

def validate(p: Params) -> bool:
    print("validating linear-rail-servo geometry ...")
    checks = []
    checks.append(("air gap positive & thin", 0.05 <= p.air_gap <= 0.5, f"{p.air_gap} mm"))
    checks.append(("carriage clears both end blocks over full travel",
                   p.carriage_x0 + p.travel + p.carriage_len / 2 < p.rail_len - p.block_x,
                   f"far edge {p.carriage_x0 + p.travel + p.carriage_len/2:.0f} < "
                   f"{p.rail_len - p.block_x:.0f} mm"))
    checks.append(("scale spans the full travel + slider footprint",
                   p.scale_len >= p.travel + p.slider_len - 1e-6,
                   f"scale {p.scale_len:.0f} vs travel+slider {p.travel + p.slider_len:.0f} mm"))
    checks.append(("slider narrower than scale (stays over the tracks)",
                   p.slider_w < p.scale_w, f"{p.slider_w} < {p.scale_w} mm"))
    checks.append(("pulley bore takes the motor shaft",
                   p.pulley_dia / 2 > p.motor_shaft_dia / 2 + 1.0,
                   f"Ø{p.pulley_dia} body vs Ø{p.motor_shaft_dia} shaft"))
    checks.append(("belt reaches both pulleys",
                   p.pulley_cx_idler > p.pulley_cx_motor,
                   f"span {p.pulley_cx_idler - p.pulley_cx_motor:.0f} mm"))
    checks.append(("rods span between end blocks",
                   p.rail_len - p.block_x > p.travel + p.carriage_len,
                   f"rod {p.rail_len - p.block_x:.0f} > {p.travel + p.carriage_len:.0f} mm"))
    checks.append(("bushing pocket fits in carriage",
                   p.bushing_od < p.carriage_h, f"Ø{p.bushing_od} < h{p.carriage_h}"))
    checks.append(("vernier absolute range covers travel",
                   p.enc.beat_len * 1e3 >= p.travel,
                   f"L_beat {p.enc.beat_len*1e3:.0f} >= travel {p.travel:.0f} mm"))
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'ok ' if passed else 'XX '}] {name:<48} {detail}")
        ok = ok and passed
    print(f"  -> {'VALID' if ok else 'INVALID'}\n")
    return ok


# --------------------------------------------------------------------------- #
# ASSEMBLY + EXPORT
# --------------------------------------------------------------------------- #

def build(p: Params, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "base": make_base(p),
        "end_block": make_end_block(p),
        "rod": make_rod(p),
        "carriage": make_carriage(p),
        "drive_pulley": make_pulley(p),
        "idler_pulley": make_pulley(p),
        "belt": make_belt(p),
        "scale_pcb": make_scale_pcb(p),
        "slider_pcb": make_slider_pcb(p),
        "motor": make_motor(p),
    }
    for name, part in parts.items():
        nsolids = len(part.solids())
        flag = "" if nsolids == 1 else f"  <-- {nsolids} solids!"
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name:<14} ({nsolids} solid){flag}")

    x0 = p.carriage_x0     # carriage at travel = 0 (home)
    placed = [
        ("base", Pos(0, 0, 0) * parts["base"]),
        ("end_block_motor", Pos(p.x_motor, 0, p.base_t) * parts["end_block"]),
        ("end_block_idler", Pos(p.x_idler, 0, p.base_t) * parts["end_block"]),
        ("rod_+", Pos(p.rail_len / 2, +p.rod_span / 2, p.rod_z) * parts["rod"]),
        ("rod_-", Pos(p.rail_len / 2, -p.rod_span / 2, p.rod_z) * parts["rod"]),
        ("scale_pcb", Pos(p.scale_x0, 0, p.scale_z) * parts["scale_pcb"]),
        ("belt", Pos(0, 0, p.rod_z - p.belt_h / 2) * parts["belt"]),
        ("drive_pulley", Pos(p.pulley_cx_motor, 0, p.rod_z - p.pulley_h / 2) * parts["drive_pulley"]),
        ("idler_pulley", Pos(p.pulley_cx_idler, 0, p.rod_z - p.pulley_h / 2) * parts["idler_pulley"]),
        ("carriage", Pos(x0, 0, 0) * parts["carriage"]),
        ("slider_pcb", Pos(x0, 0, p.slider_bottom - p.slider_t) * parts["slider_pcb"]),
        ("motor", Pos(p.pulley_cx_motor, 0, p.base_t + p.rod_z - p.pulley_h - 22.6) * parts["motor"]),
    ]
    bodies = []
    for label, body in placed:
        body.label = label
        bodies.append(body)
    export_step(Compound(children=bodies), str(outdir / "assembly.step"))
    print(f"  wrote assembly.step  ({len(bodies)} bodies)")


def report(p: Params):
    print(f"=== linear rail servo — {p.travel:.0f} mm belt axis, TT gearmotor + cap scale ===")
    print(f"rail .............. {p.rail_len:.0f} mm base, 2× Ø{p.rod_dia:.0f} rods "
          f"({p.rod_span:.0f} mm apart), carriage {p.carriage_len:.0f}×{p.carriage_w:.0f} mm")
    print(f"drive ............. TT gearmotor (vertical shaft) → GT2 pulley → belt → carriage")
    print(f"encoder ........... capacitive vernier scale in the rail: P1={p.enc.pitch1_mm} / "
          f"P2={p.enc.pitch2_mm} mm, air gap {p.air_gap} mm")
    print(f"                    {p.enc.res_fine*1e6:.1f} µm fine, absolute over "
          f"{p.enc.beat_len*1e3:.0f} mm  (see encoder.py)\n")


if __name__ == "__main__":
    p = Params()
    report(p)
    ok = validate(p)
    if ok:
        out = Path(__file__).resolve().parent / "out"
        print(f"building -> {out}")
        build(p, out)
    else:
        print("fix constraints before building.")
        sys.exit(1)
