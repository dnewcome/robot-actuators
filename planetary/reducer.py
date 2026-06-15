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

Parts (all printable + off-the-shelf): housing (mount + ring seat), ring (press-in),
carrier (planet pins + Ø8 output hub), cap (688 bearing), sun, planet x4.
Hardware: 4x Ø1.5 steel dowels (planet axles), 1x 688 bearing (8x16x5), mount screws.
"""

import sys
from dataclasses import dataclass
from math import pi, sin, cos
from pathlib import Path

# reuse the involute gear generator from the cycloidal package (shared core)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycloidal"))
from gears import spur_gear, ring_gear  # noqa: E402

from build123d import (  # noqa: E402
    BuildPart, Cylinder, Locations, PolarLocations, Pos, Mode, Align,
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
    gear_module: float = 0.75       # mm; PCD = teeth*module. 0.75 prints clean teeth.
    gear_thickness: float = 4.0     # face width
    stage_eta: float = 0.97         # single-stage mesh efficiency (machined ~0.97 / printed ~0.94)

    # --- Planet axles ----------------------------------------------------------
    planet_pin_dia: float = 1.5     # Ø1.5 steel dowel pressed into the carrier
    planet_bore_clear: float = 0.12 # planet bore radial clearance over its dowel (spins)
    ring_rim: float = 2.5           # radial body outside the ring root circle

    # --- Motor interface (2204-class drone outrunner) — VERIFY per motor -------
    motor_shaft_dia: float = 3.0
    motor_can_dia: float = 28.0
    motor_pilot_depth: float = 1.5  # recess that centers the housing on the motor can
    motor_mount_x: float = 16.0     # cross-pattern hole spacing, X pair
    motor_mount_y: float = 19.0     # cross-pattern hole spacing, Y pair
    motor_bolt_dia: float = 3.4     # M3 clearance (loose for printed holes)

    # --- Output support --------------------------------------------------------
    out_bearing_id: float = 8.0     # 688: 8 x 16 x 5
    out_bearing_od: float = 16.0
    out_bearing_w: float = 5.0
    out_hub_h: float = 7.0          # Ø8 output hub height (rides the 688)

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

    # axial datum: z=0 at the motor mount face (bottom of the flange)
    @property
    def gear_z(self) -> float:
        return self.flange_t + self.motor_face_gap

    @property
    def carrier_z(self) -> float:
        return self.gear_z + self.gear_thickness + self.gear_carrier_gap

    @property
    def housing_h(self) -> float:
        """Tube height: flange up to the top of the carrier."""
        return self.carrier_z + self.carrier_t

    @property
    def ring_seat_r(self) -> float:
        return self.ring_od / 2 + 0.10        # slip-fit bore; press or glue the ring


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

    # 6) carrier rotates inside the ring teeth without clashing
    carrier_clear = p.ring_inner_tip_r - (p.carrier_od / 2 + p.run_clear)
    check("carrier clears ring teeth", carrier_clear >= 0.4, f"{carrier_clear:.2f} mm")

    # 7) output bearing fits in the case
    check("688 bearing fits case", p.out_bearing_od + 2 * 1.5 <= p.case_od,
          f"bearing Ø{p.out_bearing_od} in case Ø{p.case_od:.1f}")

    # 8) motor mount holes land on the flange
    mount_r = max(p.motor_mount_x, p.motor_mount_y) / 2
    check("mount holes on flange", mount_r + p.motor_bolt_dia / 2 <= p.case_od / 2 - 1.0,
          f"hole edge {mount_r + p.motor_bolt_dia/2:.1f} vs flange R {p.case_od/2:.1f}")

    print("  ----")
    print(f"  RESULT: {'ALL GOOD' if ok else 'CONSTRAINTS VIOLATED'}")
    return ok


# --------------------------------------------------------------------------- #
# PART BUILDERS  (each in a local frame, base at z=0 for printing)
# --------------------------------------------------------------------------- #

def make_housing(p: ReducerParams):
    """Motor-mount flange + tube with a ring seat (ledge at the gear plane)."""
    case_r = p.case_od / 2
    with BuildPart() as h:
        Cylinder(radius=case_r, height=p.housing_h, align=_MIN)
        # wide bore from the gear plane up to the top: ring seat + carrier room
        with Locations((0, 0, p.gear_z)):
            Cylinder(radius=p.ring_seat_r, height=p.housing_h - p.gear_z + 0.1,
                     align=_MIN, mode=Mode.SUBTRACT)
        # shaft clearance through the flange (up to the ring-seat ledge)
        shaft_clear_r = p.sun_pcd / 2 + p.gear_module + p.run_clear
        Cylinder(radius=shaft_clear_r, height=p.gear_z, align=_MIN, mode=Mode.SUBTRACT)
        # pilot recess: centers the housing on the motor can
        Cylinder(radius=p.motor_can_dia / 2 + 0.2, height=p.motor_pilot_depth,
                 align=_MIN, mode=Mode.SUBTRACT)
        # motor cross-mount holes through the flange
        mount_pts = [(p.motor_mount_x / 2, 0), (-p.motor_mount_x / 2, 0),
                     (0, p.motor_mount_y / 2), (0, -p.motor_mount_y / 2)]
        with Locations(*[(x, y, 0) for x, y in mount_pts]):
            Cylinder(radius=p.motor_bolt_dia / 2, height=p.flange_t,
                     align=_MIN, mode=Mode.SUBTRACT)
        # cap screw holes (into the top rim)
        with Locations((0, 0, p.housing_h - 6.0)):
            with PolarLocations(case_r - 3.0, p.n_case_bolts):
                Cylinder(radius=p.motor_bolt_dia / 2 - 0.5, height=6.0,
                         align=_MIN, mode=Mode.SUBTRACT)
    return h.part


def make_ring(p: ReducerParams):
    """Fixed internal ring gear — prints alone (easy), presses into the housing seat."""
    return ring_gear(p.gear_module, p.n_ring, p.gear_thickness, rim=p.ring_rim)


def make_sun(p: ReducerParams):
    """Sun gear, bored for the motor shaft."""
    return spur_gear(p.gear_module, p.n_sun, p.gear_thickness, bore=p.motor_shaft_dia)


def make_planet(p: ReducerParams):
    """Planet gear, bored with running clearance over its Ø1.5 dowel."""
    return spur_gear(p.gear_module, p.n_planet, p.gear_thickness,
                     bore=p.planet_pin_dia + 2 * p.planet_bore_clear)


def make_carrier(p: ReducerParams):
    """Output carrier: holds the 4 planet dowels and carries the Ø8 output hub."""
    with BuildPart() as c:
        Cylinder(radius=p.carrier_od / 2, height=p.carrier_t, align=_MIN)
        # planet dowel holes (press fit), through the plate
        with PolarLocations(p.carrier_radius, p.n_planets):
            Cylinder(radius=p.planet_pin_dia / 2 - p.press_clear, height=p.carrier_t,
                     align=_MIN, mode=Mode.SUBTRACT)
        # central recess so the carrier clears the sun / shaft below
        sun_clear_r = p.sun_pcd / 2 + p.gear_module + p.run_clear
        Cylinder(radius=sun_clear_r, height=1.2, align=_MIN, mode=Mode.SUBTRACT)
        # Ø8 output hub on top (rides the 688 in the cap)
        with Locations((0, 0, p.carrier_t)):
            Cylinder(radius=p.out_bearing_id / 2 - p.press_clear, height=p.out_hub_h,
                     align=_MIN)
    return c.part


def make_cap(p: ReducerParams):
    """End cap: 688 bearing pocket + output through-hole, bolts to the housing."""
    case_r = p.case_od / 2
    with BuildPart() as cap:
        Cylinder(radius=case_r, height=p.cap_t + p.out_bearing_w, align=_MIN)
        # bearing pocket on the carrier-facing side (z=0)
        Cylinder(radius=p.out_bearing_od / 2 + p.press_clear, height=p.out_bearing_w,
                 align=_MIN, mode=Mode.SUBTRACT)
        # output through-hole
        Cylinder(radius=p.out_bearing_id / 2 + p.run_clear, height=p.cap_t + p.out_bearing_w,
                 align=_MIN, mode=Mode.SUBTRACT)
        # cap screw holes
        with PolarLocations(case_r - 3.0, p.n_case_bolts):
            Cylinder(radius=p.motor_bolt_dia / 2, height=p.cap_t + p.out_bearing_w,
                     align=_MIN, mode=Mode.SUBTRACT)
    return cap.part


# --------------------------------------------------------------------------- #
# ASSEMBLY + EXPORT
# --------------------------------------------------------------------------- #

def build(p: ReducerParams, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "housing": make_housing(p),
        "ring": make_ring(p),
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
    asm.append(Pos(0, 0, p.gear_z) * parts["ring"])
    asm.append(Pos(0, 0, p.gear_z) * parts["sun"])
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
