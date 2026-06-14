"""
Parametric miniature cycloidal drive for the Goolsky/Surpass 2204 1400KV outrunner.

Target: ~10:1 single stage, ~Ø34 mm, printable, feeds STEP (real CAD) + STL (print/MuJoCo).

Run:  python cycloidal/drive.py
Out:  cycloidal/out/*.step, *.stl  (+ a printed feasibility report)

Design notes
------------
- Ratio of a single-stage cycloidal (ring fixed, carrier output) = number of lobes.
  lobes = N_pins - 1.  So 10:1 -> 11 ring pins, 10-lobe disc.
- The hard part of a MINIATURE cycloidal is radial real estate, not the profile:
  the eccentric cam must hold the 3 mm motor shaft + a wall + the eccentric offset +
  a bearing, and the output-pin circle must live in the thin band between the center
  bearing OD and the lobe root. validate() reports those clearances before meshing.
- This v1 models the four kinematic parts (disc, ring, eccentric, carrier). Secondary
  features flagged TODO need the real calipered motor numbers (bolt circle, shaft length).
"""

from dataclasses import dataclass
from math import atan2, cos, sin, pi, hypot
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, BuildLine, Polyline, make_face, extrude,
    Cylinder, Hole, Locations, PolarLocations, Plane, Pos, Compound,
    Mode, export_step, export_stl, Axis,
)

from gears import spur_gear, ring_gear   # planetary involute gears

# ----------------------------------------------------------------------------- #
# PARAMETERS  — turn these knobs, re-run, read the validator.
# ----------------------------------------------------------------------------- #

@dataclass
class Params:
    # --- Reduction -------------------------------------------------------------
    # Compound, fully concentric:  motor -> PLANETARY (1st stage) -> carrier drives
    # the cycloidal ECCENTRIC -> CYCLOIDAL (2nd stage) -> output flange.
    # Total ratio = planet_ratio * cyclo_ratio. Cycloidal is LAST on purpose: it has
    # the lowest backlash and is the resolution-multiplied output stage.
    lobes: int = 10            # cycloidal stage ratio (ring fixed, carrier out). 10 -> 10:1
    # --- Cycloidal geometry ----------------------------------------------------
    pin_circle_dia: float = 32.0   # Ø of the ring-pin centers. Bigger = roomier, heavier.
    pin_dia: float = 3.0           # ring pin Ø (use Ø3 steel dowel / cut rod)
    eccentricity: float = 0.70     # E. Keep pin_R/(E*N) > 1 (smooth, no cusp); ~1.5-2 is nice.
    disc_thickness: float = 5.0
    disc_clearance: float = 0.15   # radial offset shrink of the disc profile (print fit)
    # --- Center (eccentric) bearing — sets disc center bore --------------------
    ecc_bearing_id: float = 10.0   # 6700: 10 x 15 x 4
    ecc_bearing_od: float = 15.0
    ecc_bearing_w: float = 4.0
    # --- Output (carrier) pins -------------------------------------------------
    n_out: int = 6
    out_pin_dia: float = 2.5       # Ø2.5 dowel
    out_circle_dia: float = 20.6   # must sit between center-bearing OD and lobe root
    # --- Motor interface (Goolsky 2204) — MEASURED -----------------------------
    motor_shaft_dia: float = 3.0
    motor_can_dia: float = 28.0
    # Cross mount: one pair 16 mm apart (X axis), the other 19 mm apart (Y axis), M3.
    motor_mount_x: float = 16.0       # hole-pair spacing on the X axis
    motor_mount_y: float = 19.0       # hole-pair spacing on the Y axis
    motor_bolt_dia: float = 3.4       # M3 clearance (slightly loose for printed holes)
    # --- Planetary input stage (concentric 1st reduction) ----------------------
    # Sun on the motor shaft, ring fixed to housing, carrier out drives the eccentric.
    #   planet_ratio = 1 + n_ring/n_sun,   n_ring = n_sun + 2*n_planet.
    # Runs at high speed / low torque (the efficient regime) so the cycloidal can stay
    # chunky and do the torque-dense final reduction.
    planetary: bool = True
    n_sun: int = 12            # sun teeth (12/12/36, 4 planets -> 4.0:1)
    n_planet: int = 12         # planet teeth
    n_planets: int = 4         # number of planet gears (equal-spaced)
    gear_module: float = 0.5   # mm; tooth size. PCD = teeth * module.
    planet_material: str = "machined"   # "printed" | "machined" — sets mesh efficiency
    planet_thickness: float = 4.0   # gear face width (mm)
    planet_pin_dia: float = 1.5     # planet idler pin Ø (carrier post)
    planet_ring_rim: float = 2.5    # radial rim outside the ring-gear root circle
    # backlash (lost motion) budget — sets repeatability & the servo deadband
    planet_lash_deg: float = 0.30   # lash at the planet-stage output (machined gears)
    cyclo_lash_deg: float = 0.05    # cycloidal is near-zero backlash

    # --- Housing ---------------------------------------------------------------
    housing_wall: float = 2.0

    # --- Contact strategy (THE efficiency lever): rolling vs sliding -----------
    # Ring pin / lobe contact carries the main reduction torque.
    #   "fixed"   : press-fit dowel, the lobe SLIDES on it (simple, lossy — v1)
    #   "rolling" : dowel free to spin in an oversized pocket, contact becomes ROLLING
    # Optional hardened sleeve: set pin_core_dia < pin_dia (sleeve spins on a thin core).
    pin_mode: str = "rolling"        # "fixed" | "rolling"
    pin_core_dia: float = 3.0        # steel core Ø; == pin_dia means a bare floating dowel
    pin_pocket_clear: float = 0.10   # radial clearance so a rolling dowel spins freely
    # Output pin / disc-hole contact carries output torque.
    #   "printed" : pins integral to the printed carrier (SLIDES, printed surface — v1)
    #   "steel"   : Ø out_pin_dia steel dowels pressed into the carrier (SLIDES, good surface)
    #   "bushing" : Ø out_pin_core_dia steel core + rotating bushing (ROLLING; tight at this size)
    out_mode: str = "steel"          # "printed" | "steel" | "bushing"
    out_pin_core_dia: float = 1.5    # bushing core Ø (only used in "bushing" mode)

    # --- derived ---------------------------------------------------------------
    @property
    def n_pins(self) -> int:        # ring pins
        return self.lobes + 1

    @property
    def cyclo_ratio(self) -> int:
        return self.lobes

    @property
    def n_ring(self) -> int:        # planetary internal ring teeth
        return self.n_sun + 2 * self.n_planet

    @property
    def planet_ratio(self) -> float:
        """Sun in / carrier out, ring fixed: 1 + n_ring/n_sun. 1.0 if no planetary."""
        return (1.0 + self.n_ring / self.n_sun) if self.planetary else 1.0

    @property
    def ratio(self) -> float:
        """TOTAL reduction the whole actuator delivers (this is what the sim sees)."""
        return self.planet_ratio * self.cyclo_ratio

    # planetary pitch geometry (mm)
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
    def planet_center_dist(self) -> float:
        """Sun-planet center distance = carrier radius the planets ride on."""
        return (self.sun_pcd + self.planet_pcd) / 2.0

    @property
    def output_lash_deg(self) -> float:
        """Backlash referred to the OUTPUT. The planet stage's lash divides by the
        cycloidal ratio (it's upstream); the cycloidal lash is already at the output."""
        planet_referred = (self.planet_lash_deg / self.cyclo_ratio) if self.planetary else 0.0
        return planet_referred + self.cyclo_lash_deg

    @property
    def pin_R(self) -> float:
        return self.pin_circle_dia / 2.0

    @property
    def root_R(self) -> float:
        # approximate inner (root) radius of the lobe profile
        return self.pin_R - self.pin_dia / 2.0 - 2.0 * self.eccentricity

    @property
    def housing_od(self) -> float:
        return self.pin_circle_dia + self.pin_dia + 2.0 * self.housing_wall


# ----------------------------------------------------------------------------- #
# FEASIBILITY VALIDATOR
# ----------------------------------------------------------------------------- #

def validate(p: Params) -> bool:
    R, N, E = p.pin_R, p.n_pins, p.eccentricity
    ok = True
    print(f"\n=== Cycloidal feasibility report  ({p.cyclo_ratio}:1 stage, {p.n_pins} pins) ===")
    print(f"housing OD ........ {p.housing_od:.1f} mm   (motor can Ø{p.motor_can_dia})")
    print(f"pin pitch ......... {2*pi*R/N:6.2f} mm  for Ø{p.pin_dia} pins")

    def check(label, cond, detail):
        nonlocal ok
        tag = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{tag}] {label}: {detail}")

    # 1) profile smoothness (no cusp / self-intersection)
    smooth = R / (E * N)
    check("profile smoothness  R/(E*N)", smooth > 1.0,
          f"{smooth:.2f}  (>1 required, 1.5-2 ideal)")

    # 2) eccentric cam wall: shaft bore + offset + min wall must fit in bearing ID
    cam_wall = p.ecc_bearing_id/2 - E - p.motor_shaft_dia/2
    check("eccentric thin-side wall", cam_wall >= 0.8,
          f"{cam_wall:.2f} mm  (>=0.8 mm for a printed/turned cam)")

    # 3) output pins must live between center-bearing OD and lobe root
    out_R = p.out_circle_dia / 2.0
    hole_R = (p.out_pin_dia + 2*E) / 2.0      # disc output hole radius (pin + 2E clearance)
    inner_clear = (out_R - hole_R) - p.ecc_bearing_od/2
    outer_clear = p.root_R - (out_R + hole_R)
    check("output hole clears center bearing", inner_clear >= 0.6,
          f"{inner_clear:.2f} mm gap")
    check("output hole clears lobe root", outer_clear >= 0.6,
          f"{outer_clear:.2f} mm gap")

    # 4) ring pins should be inside the housing wall
    check("ring pin inside housing", p.pin_R + p.pin_dia/2 <= p.housing_od/2,
          f"pin outer {p.pin_R + p.pin_dia/2:.1f} vs housing R {p.housing_od/2:.1f}")

    # 5) rolling-element walls (only when a sleeve / bushing is actually used)
    if p.pin_mode == "rolling" and p.pin_core_dia < p.pin_dia:
        wall = (p.pin_dia - p.pin_core_dia) / 2
        check("ring-pin sleeve wall", wall >= 0.4,
              f"{wall:.2f} mm  (Ø{p.pin_core_dia} core in Ø{p.pin_dia} sleeve)")
    if p.out_mode == "bushing":
        wall = (p.out_pin_dia - p.out_pin_core_dia) / 2
        check("output bushing wall", wall >= 0.4,
              f"{wall:.2f} mm  (Ø{p.out_pin_core_dia} core in Ø{p.out_pin_dia} bushing)")

    print(f"lobe root R ....... {p.root_R:5.2f} mm   center-bearing R {p.ecc_bearing_od/2:.2f} mm")
    ring_contact = "ROLLING" if p.pin_mode == "rolling" else "SLIDING"
    out_contact = {"printed": "SLIDING (printed)", "steel": "SLIDING (steel)",
                   "bushing": "ROLLING"}[p.out_mode]
    print(f"contact strategy .. ring pin/lobe: {ring_contact}   output pin/hole: {out_contact}")
    print("                   (rolling ring pins are the biggest single efficiency lever)")

    # --- planetary first stage --------------------------------------------------
    if p.planetary:
        print(f"\n--- Planetary input stage  ({p.planet_ratio:.2f}:1, "
              f"{p.n_sun}/{p.n_planet}/{p.n_ring} sun/planet/ring, {p.n_planets} planets) ---")
        print(f"module {p.gear_module} mm -> sun Ø{p.sun_pcd:.1f}  planet Ø{p.planet_pcd:.1f}  "
              f"ring Ø{p.ring_pcd:.1f} (PCD)")
        # a) tooth-count identity (always true by construction, but guards typos)
        check("ring = sun + 2*planet", p.n_ring == p.n_sun + 2 * p.n_planet,
              f"{p.n_ring} == {p.n_sun}+2*{p.n_planet}")
        # b) equal-spacing assembly condition: (n_sun + n_ring) divisible by n_planets
        check("equal planet spacing  (sun+ring)%planets", (p.n_sun + p.n_ring) % p.n_planets == 0,
              f"({p.n_sun}+{p.n_ring})%{p.n_planets} = {(p.n_sun + p.n_ring) % p.n_planets}")
        # c) adjacent planets must not collide: gap between planet tips > 0
        a = p.planet_center_dist
        planet_tip_dia = (p.n_planet + 2) * p.gear_module
        neighbor_gap = 2 * a * sin(pi / p.n_planets) - planet_tip_dia
        check("planets clear each other", neighbor_gap >= 0.5,
              f"{neighbor_gap:.2f} mm tip-to-tip gap")
        # d) the ring gear must fit inside the cycloidal housing
        ring_rim_R = p.ring_pcd / 2 + 1.5 * p.gear_module   # PCD/2 + a rim
        check("ring gear fits housing", ring_rim_R <= p.housing_od / 2,
              f"ring outer {ring_rim_R:.1f} vs housing R {p.housing_od/2:.1f}")
        # e) sun must bore over the motor shaft (root above shaft radius)
        sun_root_R = (p.n_sun - 2.5) * p.gear_module / 2
        check("sun clears motor shaft", sun_root_R >= p.motor_shaft_dia / 2 + 0.3,
              f"sun root R {sun_root_R:.2f} vs shaft R {p.motor_shaft_dia/2:.2f}")
        print(f"TOTAL reduction ... {p.ratio:.1f}:1   ({p.planet_ratio:.2f} x {p.cyclo_ratio})   "
              f"output backlash ~{p.output_lash_deg:.3f}°")

    print(f"\n=> {'ALL GOOD' if ok else 'HAS CONFLICTS — adjust params above'}\n")
    return ok


def hardware_bom(p: Params):
    """Non-printed parts to buy/cut for this configuration."""
    print("=== Hardware (non-printed) ===")
    if p.pin_mode == "rolling" and p.pin_core_dia < p.pin_dia:
        print(f"  ring pins:   {p.n_pins} x Ø{p.pin_core_dia} steel core + Ø{p.pin_dia} hardened sleeve (rolling)")
    else:
        kind = "free-spinning" if p.pin_mode == "rolling" else "press-fit"
        print(f"  ring pins:   {p.n_pins} x Ø{p.pin_dia} steel dowel ({kind})")
    if p.out_mode == "printed":
        print("  output pins: integral printed — upgrade to 'steel' for efficiency")
    elif p.out_mode == "steel":
        print(f"  output pins: {p.n_out} x Ø{p.out_pin_dia} steel dowel (pressed into carrier)")
    else:
        print(f"  output pins: {p.n_out} x Ø{p.out_pin_core_dia} steel core + Ø{p.out_pin_dia} bushing (rolling)")
    print(f"  bearing:     1 x {p.ecc_bearing_id:.0f}x{p.ecc_bearing_od:.0f}x{p.ecc_bearing_w:.0f} (6700-type) for the eccentric")
    if p.planetary:
        src = "printed" if p.planet_material == "printed" else "machined/MOD-gear"
        print(f"  planetary:   1 x Ø{p.sun_pcd:.0f} sun + {p.n_planets} x Ø{p.planet_pcd:.0f} planet "
              f"+ Ø{p.ring_pcd:.0f} internal ring, module {p.gear_module} ({src})")
        print(f"               {p.n_planets} x planet pin/bearing on the carrier")
    print("  screws:      4 x M3 (16 mm + 19 mm cross pattern)\n")


# ----------------------------------------------------------------------------- #
# CYCLOIDAL DISC PROFILE
# ----------------------------------------------------------------------------- #

def disc_points(p: Params, steps: int = 720):
    """Standard pin-wheel cycloidal disc profile. N = pin count, lobes = N-1."""
    R, Rr, E, N = p.pin_R, p.pin_dia/2.0, p.eccentricity, p.n_pins
    Rr_eff = Rr + p.disc_clearance     # shrink lobes slightly for print fit
    pts = []
    for i in range(steps):
        t = 2*pi*i/steps
        psi = atan2(sin((1-N)*t), (R/(E*N)) - cos((1-N)*t))
        x = R*cos(t) - Rr_eff*cos(t+psi) - E*cos(N*t)
        y = -R*sin(t) + Rr_eff*sin(t+psi) + E*sin(N*t)
        pts.append((x, y))
    return pts


# ----------------------------------------------------------------------------- #
# PARTS  (each modeled in its own frame, origin = its rotation axis)
# ----------------------------------------------------------------------------- #

def make_disc(p: Params):
    pts = disc_points(p)
    with BuildPart() as disc:
        with BuildSketch() as sk:
            with BuildLine():
                Polyline(*pts, close=True)
            make_face()
        extrude(amount=p.disc_thickness)
        # center bore for eccentric bearing OD
        Hole(radius=p.ecc_bearing_od/2)
        # output pin holes (pin Ø + 2E clearance)
        with PolarLocations(p.out_circle_dia/2, p.n_out):
            Hole(radius=(p.out_pin_dia + 2*p.eccentricity)/2)
    return disc.part


def make_ring(p: Params):
    """Pin housing + motor mount. Ring pins are separate steel dowels (BOM)."""
    h = p.disc_thickness + 2.0
    with BuildPart() as ring:
        Cylinder(radius=p.housing_od/2, height=h, align=(None, None, None))
        # cavity the disc swings in (clearance = E so the offset disc never hits the bore)
        Cylinder(radius=p.pin_R - p.pin_dia/2 + 0.2, height=h, mode=Mode.SUBTRACT)
        # ring-pin pockets (half-open toward the cavity). "rolling" oversizes the
        # pocket so the dowel spins freely (rolling lobe contact); "fixed" is press-fit.
        pocket_r = p.pin_dia/2 + (p.pin_pocket_clear if p.pin_mode == "rolling" else 0.0)
        with PolarLocations(p.pin_R, p.n_pins):
            Hole(radius=pocket_r)
        # motor mount: 2204 cross pattern — 16 mm pair on X, 19 mm pair on Y, M3
        mount_pts = [(p.motor_mount_x/2, 0), (-p.motor_mount_x/2, 0),
                     (0, p.motor_mount_y/2), (0, -p.motor_mount_y/2)]
        with Locations(*mount_pts):
            Hole(radius=p.motor_bolt_dia/2)
        # shaft / eccentric clearance through the back
        Hole(radius=p.ecc_bearing_od/2 + 0.5)
    return ring.part


def make_eccentric(p: Params):
    """Cam: 3 mm motor-shaft bore at origin, journal (bearing ID) offset by E."""
    with BuildPart() as ecc:
        with Locations((p.eccentricity, 0)):
            Cylinder(radius=p.ecc_bearing_id/2, height=p.ecc_bearing_w,
                     align=(None, None, None))
        # motor shaft bore at the true axis (origin)
        Hole(radius=p.motor_shaft_dia/2)
    return ecc.part


def make_carrier(p: Params):
    """Output flange. "printed": integral pins. "steel"/"bushing": press-fit holes for
    steel output pins (the efficiency upgrade — steel/rolling instead of a printed slide)."""
    plate_t = 3.0
    pin_len = p.disc_thickness + 1.0
    with BuildPart() as car:
        Cylinder(radius=p.root_R, height=plate_t, align=(None, None, None))
        Hole(radius=p.ecc_bearing_od/2 + 0.5)          # clear the eccentric
        if p.out_mode == "printed":
            with PolarLocations(p.out_circle_dia/2, p.n_out):
                Cylinder(radius=p.out_pin_dia/2, height=pin_len,
                         align=(None, None, None), mode=Mode.ADD)
        else:
            # press-fit holes: steel pin OD ("steel") or bushing core ("bushing")
            press_d = p.out_pin_core_dia if p.out_mode == "bushing" else p.out_pin_dia
            with PolarLocations(p.out_circle_dia/2, p.n_out):
                Hole(radius=press_d/2)
        # TODO: output shaft / hub bolt pattern once arm joint interface is decided
    return car.part


# ----------------------------------------------------------------------------- #
# PLANETARY INPUT STAGE  (sun on the motor shaft, planets on the carrier, ring fixed)
# ----------------------------------------------------------------------------- #

def make_sun(p: Params):
    """Sun gear, bored to press onto the motor shaft."""
    return spur_gear(p.gear_module, p.n_sun, p.planet_thickness, bore=p.motor_shaft_dia)


def make_planet(p: Params):
    """One planet gear, bored for its idler pin (×n_planets identical)."""
    return spur_gear(p.gear_module, p.n_planet, p.planet_thickness, bore=p.planet_pin_dia)


def make_planet_ring(p: Params):
    """Fixed internal ring gear (held in the housing)."""
    return ring_gear(p.gear_module, p.n_ring, p.planet_thickness, rim=p.planet_ring_rim)


def make_planet_carrier(p: Params):
    """Carrier plate with n_planets idler posts. Its rotation IS the planetary output;
    a center boss couples it to the cycloidal eccentric (the two stages share this part)."""
    plate_t = 2.5
    post_len = p.planet_thickness + 1.0
    a = p.planet_center_dist
    plate_r = a + (p.n_planet + 2) * p.gear_module / 2 + 1.0   # cover the planet tips
    with BuildPart() as car:
        Cylinder(radius=plate_r, height=plate_t, align=(None, None, None))
        # idler posts the planets spin on
        with PolarLocations(a, p.n_planets):
            Cylinder(radius=p.planet_pin_dia / 2, height=post_len,
                     align=(None, None, None), mode=Mode.ADD)
        # center boss bore that drives the eccentric (E offset cut on the eccentric part)
        Hole(radius=p.motor_shaft_dia / 2 + 0.4)
    return car.part


# ----------------------------------------------------------------------------- #
# ASSEMBLY + EXPORT
# ----------------------------------------------------------------------------- #

def build(p: Params, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "disc": make_disc(p),
        "ring": make_ring(p),
        "eccentric": make_eccentric(p),
        "carrier": make_carrier(p),
    }
    if p.planetary:
        parts.update({
            "sun": make_sun(p),
            "planet": make_planet(p),
            "planet_ring": make_planet_ring(p),
            "planet_carrier": make_planet_carrier(p),
        })
    for name, part in parts.items():
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name}.step / .stl")

    # nested assembly preview: disc offset by +E, parts stacked along Z so it reads as built
    z = 0.0
    asm = []
    asm.append(Pos(0, 0, z) * parts["eccentric"])
    asm.append(Pos(p.eccentricity, 0, z + 0.5) * parts["disc"])
    asm.append(Pos(0, 0, z + 0.0) * parts["ring"])
    asm.append(Pos(0, 0, z + p.disc_thickness + 1.0) * parts["carrier"])
    assembly = Compound(children=asm)
    export_step(assembly, str(outdir / "assembly.step"))
    print("  wrote assembly.step")


if __name__ == "__main__":
    p = Params()
    feasible = validate(p)
    hardware_bom(p)
    out = Path(__file__).parent / "out"
    build(p, out)
    print(f"Done -> {out}  (feasible={feasible})")
