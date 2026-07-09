"""
TWO-STAGE COMPOUND CYCLOIDAL — body-output, external-yoke straddle support.

A parametric build of the Blagojevic/Marjanovic two-stage cycloidal concept
("A New Design of a Two-Stage Cycloidal Speed Reducer", ASME J. Mech. Des. 2011):
ONE cycloid disc per stage (two discs total, not the four of a traditional
series two-stage), coupled by a shared CENTRAL DISC whose rollers thread the
windows of BOTH discs.

  stage 1 : disc1 (z1) rolls in the GROUNDED ring1  -> spins the central disc
  stage 2 : disc2 (z2) rolls in the ROTATABLE ring2 (the OUTPUT), the central
            disc now acting as the "solid support" (roles switched)

Both discs ride one common eccentric, 180 deg apart, so they balance each other.
Because both discs share the eccentric and the two rings differ by the tooth
counts, this is a COMPOUND DIFFERENTIAL, not a plain product two-stage:

        u = z1*(z2+1) / (z1 - z2)        (output co-rotates when z1 > z2)

e.g. z1=11, z2=10  ->  u = 11*11/1 = 121:1  in a single-stage-sized package.

OUTPUT SUPPORT (this variant): the output is the OUTER DRUM (ring2 body). Since
the only grounded element (ring1) sits at the INPUT end, there is no fixed part
at the far end to bolt a cap to. We therefore ground the far end from OUTSIDE:
an external YOKE (base + posts running past the drum + a far bearing carrier)
straddles the rotating drum on two bearings — moment load across the full span,
solid central input retained.

Run:  ../.venv/bin/python cycloidal-2stage/drive.py
Out:  cycloidal-2stage/out/*.step, *.stl
"""

import sys
from dataclasses import dataclass
from math import atan2, cos, sin, pi, sqrt
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, BuildLine, Polyline, make_face, extrude,
    Cylinder, Hole, Cone, Locations, PolarLocations, Pos, Compound,
    Mode, Align, export_step, export_stl,
)

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Params:
    # --- Reduction: compound differential two-stage ----------------------------
    z1: int = 11                      # stage-1 cycloid disc lobes (grounded ring1)
    z2: int = 10                      # stage-2 cycloid disc lobes (output ring2)
    pin_circle_dia: float = 40.0      # both ring-pin circles (shared scale)
    pin_dia: float = 3.0
    eccentricity: float = 0.70
    disc_thickness: float = 5.0
    disc_clearance: float = 0.15      # tooth-flank clearance
    disc_axial_clear: float = 0.30

    # --- Central-disc coupling rollers (thread BOTH discs) ---------------------
    n_roll: int = 6
    roll_circle_dia: float = 25.0
    roll_dia: float = 4.0
    cd_thickness: float = 4.0         # central plate thickness (rollers span both discs)

    # --- Eccentric cam bearings (each disc rides its cam) — 6700 10x15x4 --------
    cam_bearing_id: float = 10.0
    cam_bearing_od: float = 15.0
    cam_bearing_w: float = 4.0

    # --- Central-disc bearing (rides the eccentric mid journal) — 12x18x4 -------
    cd_bearing_id: float = 12.0
    cd_bearing_od: float = 18.0
    cd_bearing_w: float = 4.0

    # --- Input-shaft bearings (7x13x4): near in base, far in drum stub ----------
    shaft_bearing_id: float = 7.0
    shaft_bearing_od: float = 13.0
    shaft_bearing_w: float = 4.0

    # --- Output (drum) bearings — the straddle pair ----------------------------
    # near: a LARGE thin-section over the grounded ring1 OD (outboard of the gears)
    out_near_id: float = 46.0
    out_near_od: float = 54.0
    out_near_w: float = 5.0
    # far: a smaller bearing on the drum's central far stub, held by the yoke
    out_far_id: float = 20.0
    out_far_od: float = 27.0
    out_far_w: float = 5.0

    # --- Output flange (bolt the robot link here, exits beyond the yoke) --------
    out_flange_dia: float = 34.0
    out_flange_t: float = 4.0
    out_bolt_circle: float = 26.0
    out_bolt_dia: float = 3.4
    n_out_bolts: int = 4
    out_center_bore: float = 0.0

    # --- External yoke posts (ground the far bearing carrier past the drum) -----
    n_posts: int = 3
    post_dia: float = 6.0             # standoff OD
    post_bolt_dia: float = 4.0        # M4 tie-rod through the posts

    # --- Motor interface: NEMA 17 ----------------------------------------------
    motor_shaft_dia: float = 5.0
    nema_bolt_spacing: float = 31.0
    nema_pilot_dia: float = 22.0
    nema_pilot_depth: float = 2.0
    motor_bolt_dia: float = 3.4
    motor_bolt_head_dia: float = 6.0

    # --- Shell / fits ----------------------------------------------------------
    flange_t: float = 4.0
    base_gap: float = 1.0
    wall: float = 2.5
    drum_wall: float = 3.0
    press_clear: float = 0.04
    run_clear: float = 0.15

    # ---- derived: reduction ---------------------------------------------------
    @property
    def N1(self) -> int:
        return self.z1 + 1            # stage-1 ring pins
    @property
    def N2(self) -> int:
        return self.z2 + 1            # stage-2 ring pins (in the output drum)
    @property
    def ratio(self) -> float:
        return self.z1 * (self.z2 + 1) / (self.z1 - self.z2)

    @property
    def pin_R(self) -> float:
        return self.pin_circle_dia / 2.0
    @property
    def roll_R(self) -> float:
        return self.roll_circle_dia / 2.0
    @property
    def window_r(self) -> float:
        """Oversized window in each disc so the concentric central rollers clear the
        +/-E orbit (standard cycloidal pin-in-window coupling)."""
        return self.roll_dia / 2 + self.eccentricity + 0.15

    def root_R(self, z: int) -> float:
        return self.pin_R - self.pin_dia / 2.0 - 2.0 * self.eccentricity

    @property
    def gear_env_R(self) -> float:               # outermost radius of the mesh
        return self.pin_R + self.pin_dia / 2.0
    @property
    def cavity_R(self) -> float:
        return self.pin_R - self.pin_dia / 2 + self.run_clear
    @property
    def ring1_od(self) -> float:
        return self.out_near_id                  # ring1 OD is the near-bearing seat
    @property
    def drum_od(self) -> float:
        return self.out_near_od + 2 * self.drum_wall
    @property
    def post_circle_r(self) -> float:
        return self.drum_od / 2 + self.post_dia / 2 + 1.0
    @property
    def case_od(self) -> float:
        return 2 * (self.post_circle_r + self.post_dia / 2 + 1.0)
    @property
    def nema_hole_r(self) -> float:
        return self.nema_bolt_spacing / 2 * sqrt(2)

    # ---- derived: axial stack (z=0 at the motor face, motor behind) -----------
    @property
    def z_nsb(self) -> float:                    # near shaft bearing base
        return self.base_gap
    @property
    def z_d1(self) -> float:                     # disc1 base
        return self.z_nsb + self.shaft_bearing_w + 1.0
    @property
    def z_d1t(self) -> float:
        return self.z_d1 + self.disc_thickness
    @property
    def z_cd(self) -> float:                     # central plate base
        return self.z_d1t + self.disc_axial_clear
    @property
    def z_cdt(self) -> float:
        return self.z_cd + self.cd_thickness
    @property
    def z_d2(self) -> float:                     # disc2 base
        return self.z_cdt + self.disc_axial_clear
    @property
    def z_d2t(self) -> float:
        return self.z_d2 + self.disc_thickness
    @property
    def z_far(self) -> float:                    # drum body/cap base (just above disc2)
        return self.z_d2t + 1.5
    @property
    def z_capt(self) -> float:                   # drum cap top (drum is solid to here)
        return self.z_far + self.drum_wall
    @property
    def z_fob(self) -> float:                    # far output bearing base (clear above cap)
        return self.z_capt + 0.5
    @property
    def yoke_h(self) -> float:
        return self.out_far_w + 3.0
    @property
    def z_yoke(self) -> float:                   # grounded yoke plate base (= far bearing base)
        return self.z_fob
    @property
    def z_flange(self) -> float:                 # output flange base (above the yoke)
        return self.z_fob + self.yoke_h + 0.5
    @property
    def z_top(self) -> float:
        return self.z_flange + self.out_flange_t
    @property
    def z_cam1(self) -> float:
        return self.z_d1 + (self.disc_thickness - self.cam_bearing_w) / 2
    @property
    def z_cam2(self) -> float:
        return self.z_d2 + (self.disc_thickness - self.cam_bearing_w) / 2
    @property
    def z_cdb(self) -> float:
        return self.z_cd + (self.cd_thickness - self.cd_bearing_w) / 2
    @property
    def z_nob(self) -> float:                    # near output bearing base (over ring1)
        return self.z_d1
    @property
    def roll_z0(self) -> float:
        return self.z_d1 - self.disc_axial_clear
    @property
    def roll_len(self) -> float:
        return self.z_d2t + self.disc_axial_clear - self.roll_z0

    def mount_holes(self):
        h = self.nema_bolt_spacing / 2
        return [(h, h), (-h, h), (-h, -h), (h, -h)]


# --------------------------------------------------------------------------- #
# CYCLOIDAL DISC PROFILE (stage-agnostic)
# --------------------------------------------------------------------------- #

def disc_points(pin_R, pin_dia, E, N, clearance, steps: int = 720):
    R, Rr = pin_R, pin_dia / 2.0 + clearance
    pts = []
    for i in range(steps):
        t = 2 * pi * i / steps
        psi = atan2(sin((1 - N) * t), (R / (E * N)) - cos((1 - N) * t))
        x = R * cos(t) - Rr * cos(t + psi) - E * cos(N * t)
        y = -R * sin(t) + Rr * sin(t + psi) + E * sin(N * t)
        pts.append((x, y))
    return pts


# --------------------------------------------------------------------------- #
# VALIDATOR
# --------------------------------------------------------------------------- #

def validate(p: Params) -> bool:
    ok = True
    print(f"\n=== Two-stage compound cycloidal  (z1={p.z1}, z2={p.z2}) ===")
    print(f"ratio ............. {p.ratio:.2f}:1  ({'co-rotating' if p.z1>p.z2 else 'counter'})")
    print(f"drum OD ........... {p.drum_od:.1f} mm   case OD {p.case_od:.1f} mm")
    print(f"height ............ ~{p.z_top:.1f} mm + output flange (motor behind base)")

    def check(label, cond, detail):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")

    check("distinct tooth counts (differential defined)", p.z1 != p.z2,
          f"z1-z2 = {p.z1 - p.z2}")
    for z in (p.z1, p.z2):
        smooth = p.pin_R / (p.eccentricity * (z + 1))
        check(f"profile smoothness R/(E*N) z={z}", smooth > 1.0,
              f"{smooth:.2f} (>1; 1.5-2 ideal)")

    cam_wall = p.cam_bearing_id / 2 - p.eccentricity - p.motor_shaft_dia / 2
    check("eccentric cam wall", cam_wall >= 0.8, f"{cam_wall:.2f} mm (>=0.8)")

    # roller windows must clear the centre (cd bearing) and the lobe root
    inner = (p.roll_R - p.window_r) - p.cd_bearing_od / 2
    outer = p.root_R(p.z2) - (p.roll_R + p.window_r)
    check("roller window clears centre bearing", inner >= 0.5, f"{inner:.2f} mm")
    check("roller window clears lobe root", outer >= 0.5, f"{outer:.2f} mm")

    # output bearings must be OUTBOARD of the gear mesh
    check("near output bearing outboard of gears",
          p.out_near_id / 2 >= p.gear_env_R + 0.3,
          f"bore R {p.out_near_id/2:.1f} vs gear env R {p.gear_env_R:.1f}")

    # grounded ring1 wall (pins -> near-bearing seat)
    ring1_wall = p.ring1_od / 2 - p.gear_env_R
    check("ring1 wall to near-bearing seat", ring1_wall >= 1.0, f"{ring1_wall:.2f} mm")

    # drum wall over ring2 pins
    check("drum wall over ring2 pins", p.drum_od / 2 - p.gear_env_R >= 2.0,
          f"{p.drum_od/2 - p.gear_env_R:.2f} mm")

    # external posts clear the rotating drum
    check("posts clear the drum", p.post_circle_r - p.post_dia / 2 >= p.drum_od / 2 + 0.5,
          f"post inner {p.post_circle_r - p.post_dia/2:.1f} vs drum R {p.drum_od/2:.1f}")

    # straddle span between the two output bearings (moment capacity)
    span = (p.z_fob + p.out_far_w / 2) - (p.z_nob + p.out_near_w / 2)
    check("output straddle span", span >= 3 * p.out_far_w,
          f"{span:.1f} mm between near & far output bearings")

    # input journal wall over the motor shaft
    shaft_wall = p.shaft_bearing_id / 2 - p.motor_shaft_dia / 2
    check("input journal wall over shaft", shaft_wall >= 1.0, f"{shaft_wall:.2f} mm")

    check("NEMA-17 holes fit the flange",
          p.nema_hole_r + p.motor_bolt_head_dia / 2 + 0.5 <= p.case_od / 2,
          f"hole+head edge {p.nema_hole_r + p.motor_bolt_head_dia/2:.1f} vs R {p.case_od/2:.1f}")

    check("link bolts on flange",
          p.out_bolt_circle/2 + p.out_bolt_dia/2 + 1.0 <= p.out_flange_dia/2,
          f"bolt edge {p.out_bolt_circle/2 + p.out_bolt_dia/2:.1f} vs flange R {p.out_flange_dia/2:.1f}")

    print(f"\n=> {'ALL GOOD' if ok else 'HAS CONFLICTS'}\n")
    return ok


# --------------------------------------------------------------------------- #
# PARTS
# --------------------------------------------------------------------------- #

def make_eccentric(p: Params):
    """near journal (7x13) -> cam1 (+E) -> mid journal (cd bearing) -> cam2 (-E)
    -> far journal (7x13 in the drum stub). Bored to press on the motor shaft."""
    sb = p.shaft_bearing_id / 2 - p.press_clear
    cb = p.cam_bearing_id / 2 - p.press_clear
    mb = p.cd_bearing_id / 2 - p.press_clear
    with BuildPart() as e:
        # near concentric journal up to disc1
        Cylinder(radius=sb, height=p.z_d1 - p.z_nsb, align=_MIN)
        # cam1 at +E, spanning disc1
        with Locations((p.eccentricity, 0, p.z_d1 - p.z_nsb)):
            Cylinder(radius=cb, height=p.disc_thickness, align=_MIN)
        # concentric mid journal (central-disc bearing) — continuous from cam1 top to
        # cam2 base so the shaft stays one solid across the axial running clearances
        with Locations((0, 0, p.z_d1t - p.z_nsb)):
            Cylinder(radius=mb, height=p.z_d2 - p.z_d1t, align=_MIN)
        # cam2 at -E, spanning disc2
        with Locations((-p.eccentricity, 0, p.z_d2 - p.z_nsb)):
            Cylinder(radius=cb, height=p.disc_thickness, align=_MIN)
        # far concentric journal (rides the far 7x13 in the drum stub)
        with Locations((0, 0, p.z_d2t - p.z_nsb)):
            Cylinder(radius=sb, height=(p.z_far + p.shaft_bearing_w) - p.z_d2t, align=_MIN)
        Hole(radius=p.motor_shaft_dia / 2)
    return e.part


def _make_disc(p: Params, z: int):
    pts = disc_points(p.pin_R, p.pin_dia, p.eccentricity, z + 1, p.disc_clearance)
    with BuildPart() as disc:
        with BuildSketch():
            with BuildLine():
                Polyline(*pts, close=True)
            make_face()
        extrude(amount=p.disc_thickness)
        Hole(radius=p.cam_bearing_od / 2)
        with PolarLocations(p.roll_R, p.n_roll):
            Hole(radius=p.window_r)
    return disc.part


def make_disc1(p: Params):
    return _make_disc(p, p.z1)


def make_disc2(p: Params):
    return _make_disc(p, p.z2)


def make_central_disc(p: Params):
    """Mid plate on the cd bearing, with n_roll rollers that span BOTH discs
    (the 'solid support' that couples disc1's spin into disc2)."""
    with BuildPart() as c:
        # the coupler rollers (long pins threading both discs)
        with Locations((0, 0, p.roll_z0)):
            with PolarLocations(p.roll_R, p.n_roll):
                Cylinder(radius=p.roll_dia / 2, height=p.roll_len, align=_MIN)
        # the central hub/plate that carries them, on the mid journal bearing
        with Locations((0, 0, p.z_cd)):
            Cylinder(radius=p.roll_R + p.roll_dia / 2 + 1.5, height=p.cd_thickness, align=_MIN)
        Hole(radius=p.cd_bearing_od / 2)
    return c.part


def make_base(p: Params):
    """GROUNDED: NEMA-17 motor flange + near shaft-bearing hub + stationary ring1
    (pin pockets, its OD is the near-output-bearing seat) + post bosses."""
    case_r = p.case_od / 2
    bolt_r = p.motor_bolt_dia / 2
    head_r = p.motor_bolt_head_dia / 2
    cs = head_r - bolt_r
    with BuildPart() as b:
        # flange
        with Locations((0, 0, -p.flange_t)):
            Cylinder(radius=case_r, height=p.flange_t, align=_MIN)
        # one grounded puck: near hub + stationary ring1, from the flange up past disc1.
        # Solid to the ring1 OD (= near-output-bearing seat) so ring1 stays tied to ground;
        # the cavity/pins/bearing pocket are carved out below.
        Cylinder(radius=p.ring1_od / 2, height=p.z_d1t + 0.5, align=_MIN)
        # --- subtractions ---
        # near shaft-bearing pocket
        with Locations((0, 0, p.z_nsb)):
            Cylinder(radius=p.shaft_bearing_od / 2 - p.press_clear, height=p.shaft_bearing_w,
                     align=_MIN, mode=Mode.SUBTRACT)
        # motor-shaft clearance through the flange + hub floor
        with Locations((0, 0, -p.flange_t)):
            Cylinder(radius=p.motor_shaft_dia / 2 + 1.0, height=p.flange_t + p.z_nsb,
                     align=_MIN, mode=Mode.SUBTRACT)
        # cyclo cavity for disc1
        with Locations((0, 0, p.z_d1 - 0.5)):
            Cylinder(radius=p.cavity_R, height=p.disc_thickness + 1.5,
                     align=_MIN, mode=Mode.SUBTRACT)
        # ring1 pin pockets
        with Locations((0, 0, p.z_d1)):
            with PolarLocations(p.pin_R, p.N1):
                Cylinder(radius=p.pin_dia / 2, height=p.disc_thickness,
                         align=_MIN, mode=Mode.SUBTRACT)
        # NEMA holes + countersinks
        for x, y in p.mount_holes():
            with Locations((x, y, -p.flange_t)):
                Cylinder(radius=bolt_r, height=p.flange_t, align=_MIN, mode=Mode.SUBTRACT)
            with Locations((x, y, -cs)):
                Cone(bottom_radius=bolt_r, top_radius=head_r, height=cs + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        # NEMA pilot register recess
        if p.nema_pilot_depth > 0:
            with Locations((0, 0, -p.flange_t)):
                Cylinder(radius=p.nema_pilot_dia / 2 + p.run_clear, height=p.nema_pilot_depth,
                         align=_MIN, mode=Mode.SUBTRACT)
        # post tie-rod holes (through the flange, at the post circle)
        with Locations((0, 0, -p.flange_t)):
            with PolarLocations(p.post_circle_r, p.n_posts):
                Cylinder(radius=p.post_bolt_dia / 2, height=p.flange_t,
                         align=_MIN, mode=Mode.SUBTRACT)
    return b.part


def make_output_drum(p: Params):
    """ROTATING OUTPUT: stepped drum housing ring2 (over disc2), near bore riding the
    large near bearing over ring1, far central stub (far output bearing + far shaft
    bearing nested) and the output flange beyond the yoke."""
    stub_r = p.out_far_id / 2 - p.press_clear      # far output bearing rides this stub
    with BuildPart() as d:
        # drum tube + closed cap, from over ring1 (near bearing) up to the cap top
        with Locations((0, 0, p.z_nob)):
            Cylinder(radius=p.drum_od / 2, height=p.z_capt - p.z_nob, align=_MIN)
        # central far stub (carries far output bearing + the flange), clear above the cap
        with Locations((0, 0, p.z_capt)):
            Cylinder(radius=stub_r, height=p.z_flange - p.z_capt, align=_MIN)
        # output flange (above the yoke)
        with Locations((0, 0, p.z_flange)):
            Cylinder(radius=p.out_flange_dia / 2, height=p.out_flange_t, align=_MIN)
        # --- subtractions ---
        # near output bearing bore (rides over ring1)
        with Locations((0, 0, p.z_nob)):
            Cylinder(radius=p.out_near_od / 2 + p.press_clear, height=p.out_near_w,
                     align=_MIN, mode=Mode.SUBTRACT)
        # internal clearance for the ring1 body above the near bearing
        with Locations((0, 0, p.z_nob + p.out_near_w)):
            Cylinder(radius=p.ring1_od / 2 + p.run_clear,
                     height=p.z_cd - (p.z_nob + p.out_near_w) + 0.5,
                     align=_MIN, mode=Mode.SUBTRACT)
        # ring2 pin pockets over disc2
        with Locations((0, 0, p.z_d2)):
            with PolarLocations(p.pin_R, p.N2):
                Cylinder(radius=p.pin_dia / 2, height=p.disc_thickness,
                         align=_MIN, mode=Mode.SUBTRACT)
        # cyclo cavity for disc2 (leaves the cap solid above z_far)
        with Locations((0, 0, p.z_cd)):
            Cylinder(radius=p.cavity_R, height=p.z_far - p.z_cd + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        # far shaft-bearing pocket (nested in the cap/stub for the eccentric far journal)
        with Locations((0, 0, p.z_far)):
            Cylinder(radius=p.shaft_bearing_od / 2 - p.press_clear, height=p.shaft_bearing_w,
                     align=_MIN, mode=Mode.SUBTRACT)
        # link bolt holes in the flange
        with Locations((0, 0, p.z_flange)):
            with PolarLocations(p.out_bolt_circle / 2, p.n_out_bolts):
                Cylinder(radius=p.out_bolt_dia / 2, height=p.out_flange_t + 0.01,
                         align=_MIN, mode=Mode.SUBTRACT)
        if p.out_center_bore > 0:
            with Locations((0, 0, p.z_far)):
                Cylinder(radius=p.out_center_bore / 2, height=p.z_top - p.z_far,
                         align=_MIN, mode=Mode.SUBTRACT)
    return d.part


def make_yoke(p: Params):
    """GROUNDED far bearing carrier: holds the far output bearing, lets the output
    flange exit, and ties back to the base through the external posts."""
    case_r = p.case_od / 2
    with BuildPart() as y:
        with Locations((0, 0, p.z_yoke)):
            Cylinder(radius=case_r, height=p.yoke_h, align=_MIN)
        # far output bearing pocket (lower part of the plate)
        with Locations((0, 0, p.z_fob)):
            Cylinder(radius=p.out_far_od / 2 + p.press_clear, height=p.out_far_w,
                     align=_MIN, mode=Mode.SUBTRACT)
        # stub clearance above the bearing (rotating output stub passes up to the flange)
        with Locations((0, 0, p.z_fob + p.out_far_w)):
            Cylinder(radius=p.out_far_id / 2 + p.run_clear,
                     height=p.yoke_h - p.out_far_w + 0.01, align=_MIN, mode=Mode.SUBTRACT)
        # post tie-rod holes
        with Locations((0, 0, p.z_yoke)):
            with PolarLocations(p.post_circle_r, p.n_posts):
                Cylinder(radius=p.post_bolt_dia / 2, height=p.yoke_h,
                         align=_MIN, mode=Mode.SUBTRACT)
    return y.part


def make_post(p: Params):
    """One external standoff (of n_posts) grounding the yoke to the base past the drum."""
    with BuildPart() as post:
        with Locations((0, 0, 0)):
            Cylinder(radius=p.post_dia / 2, height=p.z_yoke, align=_MIN)
        Cylinder(radius=p.post_bolt_dia / 2, height=p.z_yoke, align=_MIN, mode=Mode.SUBTRACT)
    return post.part


def make_bearing(od: float, idia: float, w: float):
    outer = Cylinder(radius=od / 2, height=w, align=_MIN)
    bore = Pos(0, 0, -0.1) * Cylinder(radius=idia / 2, height=w + 0.2, align=_MIN)
    return outer - bore


def bearing_placements(p: Params):
    return [
        ("brg_out_near", p.out_near_od, p.out_near_id, p.out_near_w, (0, 0, p.z_nob)),
        ("brg_out_far",  p.out_far_od,  p.out_far_id,  p.out_far_w,  (0, 0, p.z_fob)),
        ("brg_shaft_near", p.shaft_bearing_od, p.shaft_bearing_id, p.shaft_bearing_w, (0, 0, p.z_nsb)),
        ("brg_shaft_far",  p.shaft_bearing_od, p.shaft_bearing_id, p.shaft_bearing_w, (0, 0, p.z_far)),
        ("brg_cam1", p.cam_bearing_od, p.cam_bearing_id, p.cam_bearing_w, (p.eccentricity, 0, p.z_cam1)),
        ("brg_cam2", p.cam_bearing_od, p.cam_bearing_id, p.cam_bearing_w, (-p.eccentricity, 0, p.z_cam2)),
        ("brg_central", p.cd_bearing_od, p.cd_bearing_id, p.cd_bearing_w, (0, 0, p.z_cdb)),
    ]


# --------------------------------------------------------------------------- #
# ASSEMBLY + EXPORT
# --------------------------------------------------------------------------- #

def build(p: Params, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "base": make_base(p),
        "eccentric": make_eccentric(p),
        "disc1": make_disc1(p),
        "central_disc": make_central_disc(p),
        "disc2": make_disc2(p),
        "output_drum": make_output_drum(p),
        "yoke": make_yoke(p),
        "post": make_post(p),
    }
    for name, part in parts.items():
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name}.step / .stl")

    E = p.eccentricity
    placed = [
        ("base", Pos(0, 0, 0) * parts["base"]),
        ("eccentric", Pos(0, 0, p.z_nsb) * parts["eccentric"]),
        ("disc1", Pos(E, 0, p.z_d1) * parts["disc1"]),
        ("central_disc", Pos(0, 0, 0) * parts["central_disc"]),
        ("disc2", Pos(-E, 0, p.z_d2) * parts["disc2"]),
        ("output_drum", Pos(0, 0, 0) * parts["output_drum"]),
        ("yoke", Pos(0, 0, 0) * parts["yoke"]),
    ]
    for i in range(p.n_posts):
        ang = 2 * pi * i / p.n_posts
        placed.append((f"post{i}", Pos(p.post_circle_r * cos(ang), p.post_circle_r * sin(ang), 0)
                       * parts["post"]))
    bodies = []
    for label, body in placed:
        body.label = label
        bodies.append(body)
    export_step(Compound(children=list(bodies)), str(outdir / "assembly.step"))
    print(f"  wrote assembly.step  ({len(bodies)} printed bodies)")

    for name, od, idia, w, pos in bearing_placements(p):
        body = Pos(*pos) * make_bearing(od, idia, w)
        body.label = name
        bodies.append(body)
    export_step(Compound(children=bodies), str(outdir / "assembly_full.step"))
    print(f"  wrote assembly_full.step  ({len(bodies)} bodies incl. bearings)")


def report(p: Params):
    print(f"=== {p.ratio:.1f}:1 two-stage compound cycloidal — body output, external yoke ===")
    print(f"input ............. NEMA 17 (Ø{p.motor_shaft_dia:.0f} shaft), solid central shaft")
    print(f"reduction ......... z1={p.z1}, z2={p.z2}  ->  u = z1(z2+1)/(z1-z2) = {p.ratio:.1f}:1")
    print(f"discs ............. 2 cycloid discs (one per stage), 180° opposed for balance")
    print(f"coupler ........... central disc, {p.n_roll} rollers threading both discs")
    print(f"output ............ rotating DRUM (ring2 body), Ø{p.out_flange_dia:.0f} flange")
    print(f"support ........... external yoke: base + {p.n_posts} posts + far carrier,")
    print(f"                    straddles the drum on 2 bearings (moment load)\n")


if __name__ == "__main__":
    p = Params()
    ok = validate(p)
    report(p)
    if ok:
        out = Path(__file__).resolve().parent / "out"
        print(f"building -> {out}")
        build(p, out)
    else:
        print("fix constraints before building.")
        sys.exit(1)
