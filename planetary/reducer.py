"""
STANDALONE single-stage planetary reducer for a high-RPM hobby BLDC (drone) motor.

This is the planetary stage from the hybrid compound (cycloidal/drive.py) pulled
out as its own product: sun IN (motor shaft) -> 4 planets -> FIXED ring -> carrier
OUT. One stage, ~4:1, the regime a drone motor is happiest in (high rpm, modest
reduction, low backlash). It reuses the proven involute generator in
cycloidal/gears.py and adds its own simple printable shell + motor mount.

Prototype-first: get ONE clean stage right (mesh, mount, print fits) before any
two-stage stacking — the standalone reducer IS the building block of a 2-stage.

MOTOR INTERFACE: this target motor has a drivable shaft ON THE MOUNT SIDE, so the
sun couples directly through the flange's central bore — no top-mount adapter or
prop-shaft coupling needed. The mount geometry is parametric (cross pattern +
central shaft bore + pilot recess that centers on the can); set the spacings to
the specific motor. (Note for other motors: many drone outrunners instead spin
the bell with the shaft on the PROP side, which would need an adapter.)

The sun grips the motor shaft by an interference PRESS FIT (no setscrew — the sun
is the high-rpm/low-torque input, so a press holds it with zero radial real estate);
a short collar below the gear adds press length for grip + concentricity at rpm.

Parts (all printable + off-the-shelf): housing (motor mount + RING GEAR BUILT IN),
carrier (planet pins + Ø5 D-shaft output), cap (NEMA-17 face + 625 bearing),
sun (gear + press collar), planet x4. The motor bolts underneath via 4 countersunk
M3 flat-heads (16 mm + 19 mm pairs). The OUTPUT presents as a NEMA 17: a square plate
with the 31 mm M3 pattern + a Ø5 D-shaft (no Ø22 pilot boss).
Hardware: 4x Ø1.5 steel dowels (planet axles), 1x 688 bearing (8x16x5), mount screws.
"""

import sys
from dataclasses import dataclass
from math import pi, sin, cos, sqrt, radians
from pathlib import Path

# reuse the involute gear generator from the cycloidal package (shared core)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycloidal"))
from gears import spur_gear, ring_gear  # noqa: E402

from build123d import (  # noqa: E402
    BuildPart, Box, Cone, Cylinder, Locations, PolarLocations, Pos, Mode, Align,
    export_step, export_stl,
)

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class ReducerParams:
    # --- Reduction (single planetary stage) ------------------------------------
    # planet_ratio = 1 + n_ring/n_sun,  n_ring = n_sun + 2*n_planet.
    # 12/12/36 -> 1 + 36/12 = 4.0:1 (reuses the hybrid's proven mesh).
    # For ~5:1 try n_sun=10,n_planet=15 (ring=40) -> 5.0, or n_sun=12,n_planet=18.
    n_sun: int = 12
    n_planet: int = 12
    n_planets: int = 4              # equal-spaced; needs (n_sun+n_ring) % n_planets == 0
    gear_module: float = 0.9        # mm; PCD = teeth*module. Bumped 0.75->0.9 so the
                                    # ring bore clears the motor mount-head circle (top access).
    gear_thickness: float = 4.0     # face width
    stage_eta: float = 0.97         # single-stage mesh efficiency (machined ~0.97 / printed ~0.94)

    # --- Planet axles ----------------------------------------------------------
    planet_pin_dia: float = 1.5     # Ø1.5 steel dowel pressed into the carrier
    planet_bore_clear: float = 0.12 # planet bore radial clearance over its dowel (spins)
    ring_rim: float = 2.5           # radial body outside the ring root circle

    # --- Sun shaft grip (PRESS FIT, no setscrew) -------------------------------
    # The sun is the INPUT: high rpm but low torque (peak ~88 mN·m), so an
    # interference press onto the motor shaft holds it with zero radial real estate.
    # A short collar below the gear adds press length -> grip + concentricity at rpm.
    shaft_press_fit: float = 0.05   # radial interference: bore = shaft - 2*this (printed grip)
    sun_collar_dia: float = 7.0     # press-collar OD below the gear
    sun_collar_h: float = 3.0       # collar height (set 0 for a gear-only bore)
    shaft_flat_depth: float = 0.0   # optional D-flat for anti-rotation (0 = round; >0 if D-shaft)

    # --- Motor interface (2204-class drone outrunner) — VERIFY per motor -------
    # The motor bolts UNDERNEATH the flange: M3 flat-head screws drop through
    # countersinks in the flange top and thread up into the motor's tapped holes.
    # 4 holes total = two perpendicular PAIRS: a 16 mm pair (X axis) + a 19 mm pair
    # (Y axis). The screws install from the top, dropping through the ring bore.
    motor_shaft_dia: float = 3.0
    motor_can_dia: float = 28.0
    # full-can pilot OFF by default: a Ø28 recess would straddle the mount countersinks
    # and gut the clamp material. The 4 screws locate the motor; set this >0 only with a
    # pilot_dia sized to the motor's CENTER boss (not the whole can).
    motor_pilot_depth: float = 0.0
    motor_pilot_dia: float = 12.0   # center-boss pilot Ø (only used if pilot_depth > 0)
    motor_mount_x: float = 16.0     # 16 mm pair spacing (one axis)
    motor_mount_y: float = 19.0     # 19 mm pair spacing (perpendicular axis)
    motor_bolt_dia: float = 3.4     # M3 clearance shank (loose for printed holes)
    motor_bolt_head_dia: float = 6.0  # M3 flat-head OD (sets the countersink mouth)

    # --- Output: NEMA 17 face + Ø5 D-shaft -------------------------------------
    # The output end presents like a NEMA 17 motor: a square plate with the 31 mm
    # M3 bolt pattern + a Ø5 D-shaft, so it drops into anything that mounts a NEMA 17
    # (the Ø22 pilot boss is intentionally omitted). Output rides a 625 bearing.
    out_bearing_id: float = 5.0     # 625: 5 x 16 x 5  (was 688 8x16x5)
    out_bearing_od: float = 16.0
    out_bearing_w: float = 5.0
    out_shaft_dia: float = 5.0      # NEMA 17 output shaft Ø
    out_shaft_flat_depth: float = 0.5   # D-flat depth on the protruding shaft (0=round)
    out_shaft_protrude: float = 20.0    # shaft length sticking out past the cap face
    nema_bolt_spacing: float = 31.0     # NEMA 17 bolt pattern (4x M3 on a 31 mm square)
    nema_bolt_pilot: float = 2.5    # M3 thread-form / heat-set pilot Ø
    nema_bolt_depth: float = 5.0    # blind tapped depth from the output face

    # --- Shell / fits ----------------------------------------------------------
    case_wall: float = 3.0          # radial wall outside the ring
    flange_t: float = 3.0           # motor-mount flange thickness
    cap_t: float = 3.0              # end-cap thickness (above the bearing)
    carrier_t: float = 3.0          # output carrier plate thickness
    motor_face_gap: float = 1.0     # axial gap: motor face -> gear plane
    gear_carrier_gap: float = 0.5   # axial gap: gear top -> carrier underside
    press_clear: float = 0.04       # press-fit radial clearance
    run_clear: float = 0.15         # running radial clearance
    n_case_bolts: int = 4           # cap-to-housing screws

    # backlash referred to the output (single stage -> no upstream division)
    output_lash_deg: float = 0.30

    # ---- derived geometry ---------------------------------------------------
    @property
    def n_ring(self) -> int:
        return self.n_sun + 2 * self.n_planet

    @property
    def ratio(self) -> float:
        return 1.0 + self.n_ring / self.n_sun

    @property
    def sun_pcd(self) -> float:
        return self.n_sun * self.gear_module

    @property
    def planet_pcd(self) -> float:
        return self.n_planet * self.gear_module

    @property
    def ring_pcd(self) -> float:
        return self.n_ring * self.gear_module

    @property
    def carrier_radius(self) -> float:
        """Planet-center circle radius = sun-planet center distance."""
        return (self.sun_pcd + self.planet_pcd) / 2.0

    @property
    def planet_tip_dia(self) -> float:
        return self.planet_pcd + 2 * self.gear_module        # addendum = module

    @property
    def ring_od(self) -> float:
        return self.ring_pcd + 2 * (1.25 * self.gear_module) + 2 * self.ring_rim

    @property
    def ring_inner_tip_r(self) -> float:
        """Inner radius of the ring teeth (carrier must rotate inside this)."""
        return self.ring_pcd / 2.0 - self.gear_module

    @property
    def case_od(self) -> float:
        return self.ring_od + 2 * self.case_wall

    @property
    def carrier_od(self) -> float:
        """Carrier plate Ø: reaches just past the planet pins, clears the ring teeth."""
        return 2 * (self.carrier_radius + self.planet_pin_dia / 2 + 1.0)

    @property
    def nema_plate(self) -> float:
        """Output face is a square plate covering the housing (NEMA-17-style)."""
        return self.case_od

    @property
    def out_round_len(self) -> float:
        """Round shaft length from the carrier plate top up to the cap face (rides
        the bearing); the D-flat starts above this where the shaft protrudes."""
        return 0.5 + self.cap_t + self.out_bearing_w

    @property
    def sun_pocket_r(self) -> float:
        """Bottom-face pocket so the sun gear meshes the planets with room to spare."""
        return self.sun_pcd / 2 + self.gear_module + 1.0   # sun tip + 1 mm

    def nema_holes(self):
        """4 NEMA-17 mounting holes at the corners of a 31 mm square (M3)."""
        h = self.nema_bolt_spacing / 2
        return [(h, h), (-h, h), (-h, -h), (h, -h)]

    # axial datum: z=0 at the motor mount face (bottom of the flange)
    @property
    def gear_z(self) -> float:
        return self.flange_t + self.motor_face_gap

    @property
    def sun_z(self) -> float:
        """Sun base sits a collar-height below the gear plane (collar hangs toward motor)."""
        return self.gear_z - self.sun_collar_h

    @property
    def carrier_z(self) -> float:
        return self.gear_z + self.gear_thickness + self.gear_carrier_gap

    @property
    def housing_h(self) -> float:
        """Tube height: flange up to the top of the carrier."""
        return self.carrier_z + self.carrier_t

    @property
    def ring_root_r(self) -> float:
        """Outermost radius of the ring tooth valleys (ring is built INTO the housing)."""
        return self.ring_pcd / 2 + 1.25 * self.gear_module

    @property
    def carrier_bore_r(self) -> float:
        """Plain bore above the gear plane: clears the carrier AND the ring tooth tips."""
        return max(self.carrier_od / 2 + self.run_clear, self.ring_inner_tip_r + 0.3)

    @property
    def shaft_clear_r(self) -> float:
        """Bore through the flange: clears the sun collar + motor shaft."""
        return max(self.sun_pcd / 2 + self.gear_module + self.run_clear,
                   self.sun_collar_dia / 2 + self.run_clear)

    def mount_holes(self):
        """4 motor mount holes = two perpendicular pairs: a 16 mm pair on X, a
        19 mm pair on Y. Returns (x, y, label)."""
        hx, hy = self.motor_mount_x / 2, self.motor_mount_y / 2
        return [(hx, 0, "X"), (-hx, 0, "X"), (0, hy, "Y"), (0, -hy, "Y")]


# --------------------------------------------------------------------------- #
# FEASIBILITY VALIDATOR
# --------------------------------------------------------------------------- #

def validate(p: ReducerParams) -> bool:
    ok = True
    print(f"\n=== Planetary reducer feasibility  ({p.ratio:.2f}:1, {p.n_planets} planets) ===")
    print(f"teeth  sun {p.n_sun} / planet {p.n_planet} / ring {p.n_ring}   module {p.gear_module}")
    print(f"PCD    sun Ø{p.sun_pcd:.1f} / planet Ø{p.planet_pcd:.1f} / ring Ø{p.ring_pcd:.1f}")
    print(f"case   Ø{p.case_od:.1f} x {p.housing_h:.1f} mm tall (vs motor can Ø{p.motor_can_dia})")

    def check(label, cond, detail):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")

    # 1) tooth-count consistency (geometry must close)
    check("ring = sun + 2*planet", p.n_ring == p.n_sun + 2 * p.n_planet,
          f"{p.n_ring} == {p.n_sun}+2*{p.n_planet}")

    # 2) equal-spacing assembly condition
    check("equal planet spacing  (n_sun+n_ring)%n_planets", (p.n_sun + p.n_ring) % p.n_planets == 0,
          f"({p.n_sun}+{p.n_ring}) % {p.n_planets} = {(p.n_sun + p.n_ring) % p.n_planets}")

    # 3) adjacent planets must not collide (tip circles clear along the carrier circle)
    chord = 2 * p.carrier_radius * sin(pi / p.n_planets)
    gap = chord - p.planet_tip_dia
    check("planets clear each other", gap >= 0.8, f"{gap:.2f} mm tip-to-tip gap")

    # 4) sun bore wall over the motor shaft
    sun_root_r = p.sun_pcd / 2 - 1.25 * p.gear_module
    wall = sun_root_r - p.motor_shaft_dia / 2
    check("sun bore wall over shaft", wall >= 1.2, f"{wall:.2f} mm wall (Ø{p.motor_shaft_dia} shaft)")

    # 5) ring fits inside the case wall
    case_wall_actual = (p.case_od - p.ring_od) / 2
    check("case wall around ring", case_wall_actual >= 1.5, f"{case_wall_actual:.2f} mm")

    # 5b) sun press collar: fits the flange bore, clears the motor face, walls the bore
    shaft_clear_r = p.sun_pcd / 2 + p.gear_module + p.run_clear
    check("sun collar fits flange bore", p.sun_collar_dia / 2 + p.run_clear <= shaft_clear_r,
          f"collar R {p.sun_collar_dia/2:.1f} in bore R {shaft_clear_r:.1f}")
    check("sun collar clears motor face", p.sun_z >= 0.3,
          f"collar bottom at z={p.sun_z:.1f} mm")
    bore_wall = p.sun_collar_dia / 2 - (p.motor_shaft_dia / 2 - p.shaft_press_fit)
    check("sun press-bore wall", bore_wall >= 1.0,
          f"{bore_wall:.2f} mm around Ø{p.motor_shaft_dia} shaft (press {p.shaft_press_fit*1000:.0f} µm)")

    # 6) carrier rotates inside the ring teeth without clashing
    carrier_clear = p.ring_inner_tip_r - (p.carrier_od / 2 + p.run_clear)
    check("carrier clears ring teeth", carrier_clear >= 0.4, f"{carrier_clear:.2f} mm")

    # 7) output bearing (625) fits in the cap
    check("625 bearing fits cap", p.out_bearing_od + 2 * 1.5 <= p.nema_plate,
          f"bearing Ø{p.out_bearing_od} in plate {p.nema_plate:.1f}")
    # 7b) NEMA-17 mounting holes land on the plate + clear the bearing pocket
    nr = p.nema_bolt_spacing / 2                       # hole offset on each axis
    nb_edge = nr + p.nema_bolt_pilot / 2 + 1.0         # square plate: per-axis limit
    check("NEMA holes on the plate", nb_edge <= p.nema_plate / 2,
          f"hole edge {nb_edge:.1f} vs plate half {p.nema_plate/2:.1f} (square)")
    check("NEMA holes clear bearing pocket",
          nr - p.nema_bolt_pilot / 2 >= p.out_bearing_od / 2 + p.press_clear + 1.0,
          f"hole inner {nr - p.nema_bolt_pilot/2:.1f} vs pocket R {p.out_bearing_od/2:.1f}")

    # 8) motor mount: 4 holes (16 + 19 pairs) countersunk on the flange
    head_r = p.motor_bolt_head_dia / 2
    max_r = max(sqrt(x * x + y * y) for x, y, _ in p.mount_holes())
    check("mount countersinks on flange", max_r + head_r <= p.case_od / 2 - 1.0,
          f"outer edge {max_r + head_r:.1f} vs flange R {p.case_od/2:.1f}")
    # 8b) the heads must clear the RING BORE so screws drop in from the top (the fix)
    head_clear = p.ring_inner_tip_r - (max_r + head_r)
    check("mount heads clear ring bore", head_clear >= 0.5,
          f"{head_clear:.2f} mm (head circle R {max_r + head_r:.1f} vs ring bore R {p.ring_inner_tip_r:.1f})")

    print("  ----")
    print(f"  RESULT: {'ALL GOOD' if ok else 'CONSTRAINTS VIOLATED'}")
    return ok


# --------------------------------------------------------------------------- #
# PART BUILDERS  (each in a local frame, base at z=0 for printing)
# --------------------------------------------------------------------------- #

def make_housing(p: ReducerParams):
    """Motor-mount flange + tube with the ring gear BUILT IN (teeth carved into the
    bore at the gear plane) and countersunk M3 mounts for the motor underneath."""
    case_r = p.case_od / 2
    bolt_r = p.motor_bolt_dia / 2
    head_r = p.motor_bolt_head_dia / 2
    cs_depth = head_r - bolt_r                # 90° flat-head countersink depth
    with BuildPart() as h:
        Cylinder(radius=case_r, height=p.housing_h, align=_MIN)
        # carrier-clearance bore above the gear plane (clears carrier + ring tips)
        with Locations((0, 0, p.gear_z + p.gear_thickness)):
            Cylinder(radius=p.carrier_bore_r,
                     height=p.housing_h - (p.gear_z + p.gear_thickness) + 0.1,
                     align=_MIN, mode=Mode.SUBTRACT)
        # flange bore: clears the sun collar + motor shaft (z 0..flange_t only)
        Cylinder(radius=p.shaft_clear_r, height=p.flange_t, align=_MIN, mode=Mode.SUBTRACT)
        # motor-face GAP is open, not a solid slab: clear it down to the flange so the
        # bottom is just the flange and the mount countersinks open INTO the cavity
        # (screws drop in from the top through the ring bore).
        with Locations((0, 0, p.flange_t)):
            Cylinder(radius=p.carrier_bore_r, height=p.gear_z - p.flange_t + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        # optional center-boss pilot recess (off by default; sized to the boss, not the can)
        if p.motor_pilot_depth > 0:
            Cylinder(radius=p.motor_pilot_dia / 2 + 0.2, height=p.motor_pilot_depth,
                     align=_MIN, mode=Mode.SUBTRACT)
        # motor mounts: BOTH patterns, countersunk M3 flat-heads (motor underneath)
        for x, y, _ in p.mount_holes():
            with Locations((x, y, 0)):
                Cylinder(radius=bolt_r, height=p.flange_t, align=_MIN, mode=Mode.SUBTRACT)
            with Locations((x, y, p.flange_t - cs_depth)):     # cone opens at the top face
                Cone(bottom_radius=bolt_r, top_radius=head_r, height=cs_depth + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        # cap screw holes (into the top rim)
        with Locations((0, 0, p.housing_h - 6.0)):
            with PolarLocations(case_r - 3.0, p.n_case_bolts):
                Cylinder(radius=bolt_r - 0.5, height=6.0, align=_MIN, mode=Mode.SUBTRACT)
    hp = h.part
    # carve the internal ring teeth at the gear plane: subtract the toothed-bore
    # NEGATIVE (a root-radius disc minus the ring annulus) so the teeth remain,
    # integral to the housing wall.
    gt = p.gear_thickness
    ring_neg = (Pos(0, 0, gt / 2) * Cylinder(radius=p.ring_root_r, height=gt)
                - ring_gear(p.gear_module, p.n_ring, gt, rim=p.ring_rim))
    return hp - Pos(0, 0, p.gear_z) * ring_neg


def make_sun(p: ReducerParams):
    """Sun gear + short press collar, bored as an interference PRESS FIT onto the
    motor shaft (no setscrew). Optional D-flat (shaft_flat_depth>0) adds positive
    anti-rotation for a D-shaft. Collar hangs toward the motor; gear is on top."""
    bore_r = p.motor_shaft_dia / 2.0 - p.shaft_press_fit     # undersized -> grips
    H = p.sun_collar_h + p.gear_thickness
    sun = Pos(0, 0, p.sun_collar_h) * spur_gear(
        p.gear_module, p.n_sun, p.gear_thickness, bore=0.0)
    if p.sun_collar_h > 0:
        cr = p.sun_collar_dia / 2.0
        sun = sun + Pos(0, 0, p.sun_collar_h / 2.0) * Cylinder(radius=cr, height=p.sun_collar_h)
    # bore (centered, tall enough to pass fully through the part)
    bore = Cylinder(radius=bore_r, height=3 * H)
    if p.shaft_flat_depth > 0:                                # carve a flat -> D-bore
        flat_pos = bore_r - p.shaft_flat_depth
        bore = bore - Pos(flat_pos + bore_r, 0, 0) * Box(2 * bore_r, 4 * bore_r, 3 * H)
    return sun - bore


def make_planet(p: ReducerParams):
    """Planet gear, bored with running clearance over its Ø1.5 dowel."""
    return spur_gear(p.gear_module, p.n_planet, p.gear_thickness,
                     bore=p.planet_pin_dia + 2 * p.planet_bore_clear)


def make_carrier(p: ReducerParams):
    """Output carrier: holds the 4 planet dowels, opens a sun-clearance pocket on the
    bottom so the sun meshes the planets, and carries the Ø5 D-shaft output up top."""
    shaft_len = p.out_round_len + p.out_shaft_protrude
    sd = p.out_shaft_dia
    with BuildPart() as c:
        Cylinder(radius=p.carrier_od / 2, height=p.carrier_t, align=_MIN)
        # planet dowel holes (press fit), through the plate
        with PolarLocations(p.carrier_radius, p.n_planets):
            Cylinder(radius=p.planet_pin_dia / 2 - p.press_clear, height=p.carrier_t,
                     align=_MIN, mode=Mode.SUBTRACT)
        # sun-clearance pocket on the bottom (so the sun gear has room to mesh)
        Cylinder(radius=p.sun_pocket_r, height=2.0, align=_MIN, mode=Mode.SUBTRACT)
        # Ø5 output shaft on top: round through the bearing, D-flat on the protrusion
        with Locations((0, 0, p.carrier_t)):
            Cylinder(radius=sd / 2, height=shaft_len, align=_MIN)
    shaft = c.part
    if p.out_shaft_flat_depth > 0:                 # cut the D-flat on the protruding part
        flat_pos = sd / 2 - p.out_shaft_flat_depth
        z0 = p.carrier_t + p.out_round_len         # flat starts at the cap face
        shaft = shaft - (Pos(flat_pos + sd, 0, z0 + p.out_shaft_protrude / 2)
                         * Box(2 * sd, 4 * sd, p.out_shaft_protrude + 0.2))
    return shaft


def make_cap(p: ReducerParams):
    """NEMA-17 output face: square plate + 31 mm M3 bolt pattern + 625 bearing pocket
    + Ø5 shaft through-hole. Bolts to the housing; no Ø22 pilot boss (by request)."""
    plate = p.nema_plate
    th = p.cap_t + p.out_bearing_w
    with BuildPart() as cap:
        Box(plate, plate, th, align=_MIN)
        # 625 bearing pocket on the carrier-facing side (z=0)
        Cylinder(radius=p.out_bearing_od / 2 + p.press_clear, height=p.out_bearing_w,
                 align=_MIN, mode=Mode.SUBTRACT)
        # Ø5 output through-hole
        Cylinder(radius=p.out_shaft_dia / 2 + p.run_clear, height=th,
                 align=_MIN, mode=Mode.SUBTRACT)
        # NEMA-17 mounting holes: blind M3, tapped from the OUTPUT face (top)
        for x, y in p.nema_holes():
            with Locations((x, y, th - p.nema_bolt_depth)):
                Cylinder(radius=p.nema_bolt_pilot / 2, height=p.nema_bolt_depth + 0.01,
                         align=_MIN, mode=Mode.SUBTRACT)
        # cap-to-housing screws (clearance), on-axis so they miss the NEMA corner holes
        with PolarLocations(p.case_od / 2 - 3.0, p.n_case_bolts):
            Cylinder(radius=p.motor_bolt_dia / 2, height=th, align=_MIN, mode=Mode.SUBTRACT)
    return cap.part


# --------------------------------------------------------------------------- #
# ASSEMBLY + EXPORT
# --------------------------------------------------------------------------- #

def build(p: ReducerParams, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "housing": make_housing(p),     # ring gear is built into the housing now
        "sun": make_sun(p),
        "planet": make_planet(p),
        "carrier": make_carrier(p),
        "cap": make_cap(p),
    }
    for name, part in parts.items():
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name}.step / .stl")

    # assemble on the shared axial datum (z=0 at the motor face)
    asm = [Pos(0, 0, 0) * parts["housing"]]
    asm.append(Pos(0, 0, p.sun_z) * parts["sun"])
    for i in range(p.n_planets):
        ang = 2 * pi * i / p.n_planets
        x, y = p.carrier_radius * cos(ang), p.carrier_radius * sin(ang)
        asm.append(Pos(x, y, p.gear_z) * parts["planet"])
    asm.append(Pos(0, 0, p.carrier_z) * parts["carrier"])
    asm.append(Pos(0, 0, p.housing_h + 0.5) * parts["cap"])
    from build123d import Compound
    Compound(children=asm)  # validates placement; STEP of the stack:
    export_step(Compound(children=[a for a in asm]), str(outdir / "assembly.step"))
    print(f"  wrote assembly.step")


def report(p: ReducerParams):
    """Torque / speed for the 2204 driving this single stage (matches mujoco/actuator.py)."""
    kv, volt, i_burst = 1400.0, 11.1, 13.0
    kt = 60.0 / (2 * pi * kv)                      # N·m/A
    motor_nl = kv * volt                            # rpm, no load
    out_nl = motor_nl / p.ratio
    peak_motor = kt * i_burst
    peak_out = peak_motor * p.ratio * p.stage_eta
    print(f"\n=== 2204 (1400KV, {volt}V) + {p.ratio:.0f}:1 single planetary stage ===")
    print(f"no-load speed ..... {motor_nl:.0f} rpm motor  ->  {out_nl:.0f} rpm output")
    print(f"peak torque ....... {peak_motor*1000:.1f} mN·m motor -> {peak_out*1000:.0f} mN·m output "
          f"(@ {i_burst:.0f} A, η {p.stage_eta:.0%})")
    print(f"backlash .......... ~{p.output_lash_deg:.2f}° at output (single stage, undivided)")
    print(f"reflected inertia . scales x{p.ratio**2:.0f} (rotor J seen at output)")
    grip = "round" if p.shaft_flat_depth == 0 else f"D-flat {p.shaft_flat_depth:.1f}mm"
    print(f"sun grip .......... press fit Ø{p.motor_shaft_dia - 2*p.shaft_press_fit:.2f} bore "
          f"({p.shaft_press_fit*1000:.0f} µm interference, {grip}), {p.sun_collar_h:.0f}mm collar, no setscrew")
    print(f"output ............ NEMA-17 face ({p.nema_plate:.0f}mm sq, {p.nema_bolt_spacing:.0f}mm M3 pattern) "
          f"+ Ø{p.out_shaft_dia:.0f} D-shaft ({p.out_shaft_protrude:.0f}mm out), 625 bearing")


if __name__ == "__main__":
    p = ReducerParams()
    ok = validate(p)
    report(p)
    if ok:
        out = Path(__file__).resolve().parent / "out"
        print(f"\nbuilding STLs -> {out}")
        build(p, out)
    else:
        print("\nfix constraints before building.")
        sys.exit(1)
