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

# ----------------------------------------------------------------------------- #
# PARAMETERS  — turn these knobs, re-run, read the validator.
# ----------------------------------------------------------------------------- #

@dataclass
class Params:
    # --- Reduction -------------------------------------------------------------
    lobes: int = 10            # = reduction ratio (ring fixed, carrier out). 10 -> 10:1
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
    # --- Housing ---------------------------------------------------------------
    housing_wall: float = 2.0

    # --- derived ---------------------------------------------------------------
    @property
    def n_pins(self) -> int:        # ring pins
        return self.lobes + 1

    @property
    def ratio(self) -> int:
        return self.lobes

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
    print(f"\n=== Cycloidal feasibility report  ({p.ratio}:1, {p.n_pins} pins) ===")
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

    print(f"lobe root R ....... {p.root_R:5.2f} mm   center-bearing R {p.ecc_bearing_od/2:.2f} mm")
    print(f"=> {'ALL GOOD' if ok else 'HAS CONFLICTS — adjust params above'}\n")
    return ok


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
        # ring-pin pockets (half-open toward the cavity)
        with PolarLocations(p.pin_R, p.n_pins):
            Hole(radius=p.pin_dia/2)
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
    """Output flange + integral output pins that pass through the disc holes."""
    plate_t = 3.0
    pin_len = p.disc_thickness + 1.0
    with BuildPart() as car:
        Cylinder(radius=p.root_R, height=plate_t, align=(None, None, None))
        Hole(radius=p.ecc_bearing_od/2 + 0.5)          # clear the eccentric
        with PolarLocations(p.out_circle_dia/2, p.n_out):
            Cylinder(radius=p.out_pin_dia/2, height=pin_len,
                     align=(None, None, None), mode=Mode.ADD)
        # TODO: output shaft / hub bolt pattern once arm joint interface is decided
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
    out = Path(__file__).parent / "out"
    build(p, out)
    print(f"Done -> {out}  (feasible={feasible})")
