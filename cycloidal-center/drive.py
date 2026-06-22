"""
STRADDLE-CARRIER cycloidal drive — a moment-load-capable robot-joint output.

The usual hobby cycloidal carries the output pins on ONE plate (cantilever) and necks
to a single small output bearing — almost no moment/axial capacity. This design takes
the output from a TWO-PART PIN CARRIER: a plate on each side of the disc, BOLTED
TOGETHER BY THE OUTPUT PINS into one rigid cage, supported by a bearing at BOTH ends.

Fully symmetric, nested bearing stack (4 support bearings + 1 disc bearing):
  - 2x 7x13x4  : support the eccentric INPUT shaft, one journal at each end
  - 2x 30x37x4 : support the OUTPUT carrier cage, one at each end (the bigger,
                 concentric "out" pair that carries the overhung moment)
  - 1x 6700 (10x15x4) : the disc rides the eccentric cam on this
At each end a small 7x13 sits CONCENTRICALLY INSIDE a large 30x37 — input-shaft bearing
nested in the carrier bearing — so the reduction path and the load path are separate.

Layout (z=0 at the motor face, motor behind the base):
    motor ─ flange ─[30x37 + 7x13]─ carrier_back ─┐
                          cam ─[6700]─ disc        │  output pins (roller + clamp
                                       carrier_front┘  screw) span the disc and bolt
                          ─[7x13 + 30x37]─ OUTPUT FLANGE   the two plates into a cage

Single stage, ratio = lobes. A planetary front stage can be added later (compound).

Run:  ../.venv/bin/python cycloidal-center/drive.py
Out:  cycloidal-center/out/*.step, *.stl
"""

import sys
from dataclasses import dataclass
from math import atan2, cos, sin, pi, sqrt
from pathlib import Path

from build123d import (
    BuildPart, BuildSketch, BuildLine, Polyline, make_face, extrude,
    Box, Cylinder, Hole, Cone, Locations, PolarLocations, Pos, Compound,
    Mode, Align, export_step, export_stl,
)

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class Params:
    # --- Reduction (single-stage cycloidal: ring fixed, carrier out) ------------
    lobes: int = 10                 # ratio = lobes (11 ring pins)
    pin_circle_dia: float = 35.0
    pin_dia: float = 3.0
    eccentricity: float = 0.70
    disc_thickness: float = 5.0
    disc_clearance: float = 0.15
    disc_axial_clear: float = 0.30

    # --- Disc / eccentric cam bearing (disc rides the cam) ---------------------
    ecc_bearing_id: float = 10.0    # 6700: 10 x 15 x 4
    ecc_bearing_od: float = 15.0
    ecc_bearing_w: float = 4.0

    # --- Input-shaft bearings (7x13x4, ONE AT EACH END) ------------------------
    # The eccentric has a concentric journal at each end riding one of these; nested
    # inside the carrier hubs. The motor no longer cantilevers the eccentric.
    shaft_bearing_id: float = 7.0   # MR137: 7 x 13 x 4 (x2)
    shaft_bearing_od: float = 13.0
    shaft_bearing_w: float = 4.0

    # --- Carrier (output) bearings — the LARGER pair, ONE AT EACH END ----------
    carrier_bearing_id: float = 30.0  # 30 x 37 x 4 thin-section (x2)
    carrier_bearing_od: float = 37.0
    carrier_bearing_w: float = 4.0

    # --- Output pins (steel roller sleeve + clamp screw = the cage fastener) ----
    n_out: int = 6
    out_circle_dia: float = 22.0
    out_sleeve_od: float = 4.0      # steel roller the disc rides on
    out_screw_dia: float = 2.5      # M2.5 clamp screw down the sleeve
    out_screw_pilot: float = 2.05   # thread/heat-set pilot in carrier_front
    out_screw_head_dia: float = 4.8 # counterbore in carrier_back
    out_screw_head_h: float = 2.5

    # --- Output flange (bolt the robot link here) ------------------------------
    out_flange_dia: float = 36.0
    out_flange_t: float = 3.0
    out_bolt_circle: float = 28.0
    out_bolt_dia: float = 3.4
    n_out_bolts: int = 4
    out_center_bore: float = 0.0

    # --- Motor interface: NEMA 17 mount (stepper / NEMA-mount motor) -----------
    # Ø5 shaft, 31 mm square M3 bolt pattern (holes at ±15.5), Ø22 pilot register.
    motor_shaft_dia: float = 5.0      # NEMA 17 shaft Ø5 (the eccentric presses onto this)
    nema_bolt_spacing: float = 31.0   # 31 mm square M3 pattern
    nema_pilot_dia: float = 22.0      # Ø22 register boss on the NEMA face
    nema_pilot_depth: float = 2.0     # flange recess to receive it (0 = none)
    motor_bolt_dia: float = 3.4       # M3 clearance
    motor_bolt_head_dia: float = 6.0  # M3 flat-head OD (countersink mouth)

    # --- Shell / fits ----------------------------------------------------------
    housing_wall: float = 2.0
    flange_t: float = 3.0
    cap_t: float = 3.0
    base_gap: float = 1.0
    carrier_t: float = 3.0
    press_clear: float = 0.04
    run_clear: float = 0.15
    n_case_bolts: int = 4

    # ---- derived: reduction / profile -----------------------------------------
    @property
    def n_pins(self) -> int:
        return self.lobes + 1

    @property
    def ratio(self) -> float:
        return float(self.lobes)

    @property
    def pin_R(self) -> float:
        return self.pin_circle_dia / 2.0

    @property
    def root_R(self) -> float:
        return self.pin_R - self.pin_dia / 2.0 - 2.0 * self.eccentricity

    @property
    def out_hole_r(self) -> float:
        return self.out_sleeve_od / 2 + self.eccentricity + 0.15

    @property
    def sleeve_len(self) -> float:
        return self.disc_thickness + 2 * self.disc_axial_clear

    @property
    def housing_od(self) -> float:
        return self.pin_circle_dia + self.pin_dia + 2.0 * self.housing_wall

    @property
    def cap_bolt_r(self) -> float:
        """Cap/housing bolt circle — outboard of the big carrier bearing pocket."""
        return self.carrier_bearing_od / 2 + 2.5

    @property
    def nema_hole_r(self) -> float:
        """Radius of a NEMA-17 mount hole (corner of the 31 mm square)."""
        return self.nema_bolt_spacing / 2 * sqrt(2)

    @property
    def case_od(self) -> float:
        return max(self.housing_od, self.out_flange_dia + 2 * self.housing_wall,
                   2 * (self.cap_bolt_r + self.motor_bolt_dia / 2 + 1.5),
                   2 * (self.nema_hole_r + self.motor_bolt_head_dia / 2 + 1.0))

    @property
    def cyclo_cavity_r(self) -> float:
        return self.pin_R - self.pin_dia / 2 + self.run_clear

    @property
    def carrier_hub_dia(self) -> float:
        return self.carrier_bearing_id

    # ---- derived: axial stack (z=0 at the motor face) -------------------------
    @property
    def z_bb(self) -> float:               # back carrier bearing base
        return self.base_gap
    @property
    def z_cback(self) -> float:            # carrier_back plate base
        return self.z_bb + self.carrier_bearing_w
    @property
    def z_disc(self) -> float:
        return self.z_cback + self.carrier_t
    @property
    def z_disc_top(self) -> float:
        return self.z_disc + self.disc_thickness
    @property
    def z_cfront(self) -> float:           # carrier_front plate base
        return self.z_disc_top
    @property
    def z_fb(self) -> float:               # front carrier bearing base
        return self.z_cfront + self.carrier_t
    @property
    def z_flange_out(self) -> float:       # output flange base
        return self.z_fb + self.carrier_bearing_w
    @property
    def z_split(self) -> float:
        return self.z_cfront
    @property
    def z_cap_top(self) -> float:
        return self.z_flange_out + self.out_flange_t
    @property
    def z_cam(self) -> float:              # cam journal (disc bearing) base
        return self.z_disc + (self.disc_thickness - self.ecc_bearing_w) / 2

    def mount_holes(self):
        """NEMA 17: 4 holes at the corners of a 31 mm square."""
        h = self.nema_bolt_spacing / 2
        return [(h, h), (-h, h), (-h, -h), (h, -h)]


# --------------------------------------------------------------------------- #
# CYCLOIDAL DISC PROFILE
# --------------------------------------------------------------------------- #

def disc_points(p: Params, steps: int = 720):
    R, Rr, E, N = p.pin_R, p.pin_dia / 2.0, p.eccentricity, p.n_pins
    Rr_eff = Rr + p.disc_clearance
    pts = []
    for i in range(steps):
        t = 2 * pi * i / steps
        psi = atan2(sin((1 - N) * t), (R / (E * N)) - cos((1 - N) * t))
        x = R * cos(t) - Rr_eff * cos(t + psi) - E * cos(N * t)
        y = -R * sin(t) + Rr_eff * sin(t + psi) + E * sin(N * t)
        pts.append((x, y))
    return pts


# --------------------------------------------------------------------------- #
# FEASIBILITY VALIDATOR
# --------------------------------------------------------------------------- #

def validate(p: Params) -> bool:
    ok = True
    R, N, E = p.pin_R, p.n_pins, p.eccentricity
    print(f"\n=== Straddle-carrier cycloidal  ({p.lobes}:1, {p.n_pins} pins) ===")
    print(f"housing OD ........ {p.housing_od:.1f} mm   case OD {p.case_od:.1f} mm")
    print(f"height ............ ~{p.z_cap_top:.1f} mm + output flange (motor behind base)")

    def check(label, cond, detail):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")

    smooth = R / (E * N)
    check("profile smoothness R/(E*N)", smooth > 1.0, f"{smooth:.2f} (>1; 1.5-2 ideal)")

    cam_wall = p.ecc_bearing_id / 2 - E - p.motor_shaft_dia / 2
    check("eccentric cam wall", cam_wall >= 0.8, f"{cam_wall:.2f} mm (>=0.8)")

    out_R = p.out_circle_dia / 2
    inner = (out_R - p.out_hole_r) - p.ecc_bearing_od / 2
    outer = p.root_R - (out_R + p.out_hole_r)
    check("output hole clears centre bearing", inner >= 0.5, f"{inner:.2f} mm")
    check("output hole clears lobe root", outer >= 0.5, f"{outer:.2f} mm")

    check("ring pin inside housing", p.pin_R + p.pin_dia / 2 <= p.housing_od / 2,
          f"pin outer {p.pin_R + p.pin_dia/2:.1f} vs housing R {p.housing_od/2:.1f}")

    # STRADDLE carrier: both ends on big bearings, small shaft bearings nested inside
    check("carrier bearings fit case", p.carrier_bearing_od + 2 * 2.0 <= p.case_od,
          f"bearing Ø{p.carrier_bearing_od} in case Ø{p.case_od:.1f}")
    check("shaft bearing nests in carrier hub",
          p.shaft_bearing_od + 2 * 1.5 <= p.carrier_hub_dia,
          f"7x13 Ø{p.shaft_bearing_od} inside hub Ø{p.carrier_hub_dia}")
    shaft_wall = p.shaft_bearing_id / 2 - p.motor_shaft_dia / 2
    check("input journal wall over shaft", shaft_wall >= 1.0,
          f"{shaft_wall:.2f} mm (Ø{p.shaft_bearing_id} journal over Ø{p.motor_shaft_dia} shaft)")
    check("NEMA-17 holes fit the flange",
          p.nema_hole_r + p.motor_bolt_head_dia / 2 + 0.5 <= p.case_od / 2,
          f"hole+head edge {p.nema_hole_r + p.motor_bolt_head_dia/2:.1f} vs flange R {p.case_od/2:.1f}")
    sleeve_wall = (p.out_sleeve_od - p.out_screw_dia) / 2
    check("output roller wall over screw", sleeve_wall >= 0.5,
          f"{sleeve_wall:.2f} mm (Ø{p.out_sleeve_od} roller over M{p.out_screw_dia})")
    span = 2 * p.carrier_bearing_w + p.disc_thickness + 2 * p.carrier_t
    check("carrier moment span", span >= 3 * p.carrier_bearing_w,
          f"{span:.1f} mm between the two carrier bearings")

    check("link bolts on flange",
          p.out_bolt_circle/2 + p.out_bolt_dia/2 + 1.0 <= p.out_flange_dia/2,
          f"bolt edge {p.out_bolt_circle/2 + p.out_bolt_dia/2:.1f} vs flange R {p.out_flange_dia/2:.1f}")
    check("cap bolts clear carrier bearing",
          p.cap_bolt_r - p.motor_bolt_dia/2 >= p.carrier_bearing_od/2 + 0.3,
          f"cap bolt inner {p.cap_bolt_r - p.motor_bolt_dia/2:.1f} vs bearing R {p.carrier_bearing_od/2:.1f}")

    print(f"lobe root R ....... {p.root_R:.2f} mm   out-pin R {p.out_circle_dia/2:.2f} "
          f"(hole r {p.out_hole_r:.2f})   centre-brg R {p.ecc_bearing_od/2:.2f}")
    print(f"\n=> {'ALL GOOD' if ok else 'HAS CONFLICTS'}\n")
    return ok


# --------------------------------------------------------------------------- #
# PARTS
# --------------------------------------------------------------------------- #

def make_eccentric(p: Params):
    """Concentric back journal (rides the back 7x13) -> offset cam (disc bearing) ->
    concentric front journal (rides the front 7x13). Bored to press on the motor shaft."""
    sb = p.shaft_bearing_id / 2 - p.press_clear
    z_back0 = p.z_bb                                  # back journal base
    z_camb = p.z_cam                                  # cam base
    z_front_top = p.z_fb + p.shaft_bearing_w          # front journal top
    with BuildPart() as e:
        # back concentric journal + body up to the cam
        Cylinder(radius=sb, height=z_camb - z_back0, align=_MIN)
        # offset cam journal (disc bearing rides this)
        with Locations((p.eccentricity, 0, z_camb - z_back0)):
            Cylinder(radius=p.ecc_bearing_id / 2 - p.press_clear, height=p.ecc_bearing_w,
                     align=_MIN)
        # concentric front journal up through the front 7x13
        with Locations((0, 0, z_camb - z_back0 + p.ecc_bearing_w)):
            Cylinder(radius=sb, height=z_front_top - (z_camb + p.ecc_bearing_w), align=_MIN)
        Hole(radius=p.motor_shaft_dia / 2)            # press onto the motor shaft
    return e.part


def make_disc(p: Params):
    pts = disc_points(p)
    with BuildPart() as disc:
        with BuildSketch():
            with BuildLine():
                Polyline(*pts, close=True)
            make_face()
        extrude(amount=p.disc_thickness)
        Hole(radius=p.ecc_bearing_od / 2)
        with PolarLocations(p.out_circle_dia / 2, p.n_out):
            Hole(radius=p.out_hole_r)
    return disc.part


def _carrier_plate(p: Params, mode: str):
    """Two-part carrier plate, symmetric. Each has a hub (into its 30x37 carrier bearing)
    with a nested 7x13 pocket for the input-shaft journal. 'back' = hub down, screw heads;
    'front' = hub up + output flange, screw pilots, link bolts."""
    hub_dia = p.carrier_hub_dia
    plate_r = max(p.out_circle_dia / 2 + p.out_hole_r + 2.0, hub_dia / 2 + 1.0)
    bw = p.carrier_bearing_w
    with BuildPart() as c:
        Cylinder(radius=plate_r, height=p.carrier_t, align=_MIN)
        if mode == "back":
            with Locations((0, 0, -bw)):
                Cylinder(radius=hub_dia / 2, height=bw, align=_MIN)
        else:
            with Locations((0, 0, p.carrier_t)):
                Cylinder(radius=hub_dia / 2, height=bw, align=_MIN)
                with Locations((0, 0, bw)):
                    Cylinder(radius=p.out_flange_dia / 2, height=p.out_flange_t, align=_MIN)
    body = c.part
    sbore = p.shaft_bearing_od / 2 - p.press_clear     # 7x13 press pocket
    with BuildPart() as cut:
        if mode == "back":
            # nested 7x13 pocket in the hub (input-shaft journal rides it)
            with Locations((0, 0, -bw)):
                Cylinder(radius=sbore, height=bw, align=_MIN)
            # clearance through the plate for the eccentric journal — sized to the
            # journal (not the disc bearing) so the plate forms a shoulder that SEATS
            # the 7x13 below it instead of letting it float
            Cylinder(radius=p.shaft_bearing_id / 2 + p.run_clear, height=p.carrier_t + 0.02,
                     align=_MIN)
            # output-screw clearance + head counterbore (heads on the back face)
            with PolarLocations(p.out_circle_dia / 2, p.n_out):
                Cylinder(radius=p.out_screw_dia / 2 + 0.2, height=p.carrier_t, align=_MIN)
            with Locations((0, 0, -bw)):
                with PolarLocations(p.out_circle_dia / 2, p.n_out):
                    Cylinder(radius=p.out_screw_head_dia / 2, height=p.out_screw_head_h,
                             align=_MIN)
        else:
            # nested 7x13 pocket in the up-hub
            with Locations((0, 0, p.carrier_t)):
                Cylinder(radius=sbore, height=bw, align=_MIN)
            # clearance through the plate for the front concentric journal
            Cylinder(radius=p.shaft_bearing_id / 2 + p.run_clear, height=p.carrier_t + 0.02,
                     align=_MIN)
            # output-screw pilots (thread/heat-set)
            with PolarLocations(p.out_circle_dia / 2, p.n_out):
                Cylinder(radius=p.out_screw_pilot / 2, height=p.carrier_t, align=_MIN)
            # link bolt holes in the flange
            with Locations((0, 0, p.carrier_t + bw)):
                with PolarLocations(p.out_bolt_circle / 2, p.n_out_bolts):
                    Cylinder(radius=p.out_bolt_dia / 2, height=p.out_flange_t + 0.02, align=_MIN)
            if p.out_center_bore > 0:
                Cylinder(radius=p.out_center_bore / 2,
                         height=p.carrier_t + bw + p.out_flange_t + 0.02, align=_MIN)
    return body - cut.part


def make_carrier_back(p: Params):
    return _carrier_plate(p, "back")


def make_carrier_front(p: Params):
    return _carrier_plate(p, "front")


def make_housing(p: Params):
    """Motor flange + cyclo tube with fixed ring-pin pockets + the back carrier-bearing
    pocket low in the base. Tops out where the output cap bolts on."""
    case_r = p.case_od / 2
    z_top = p.z_split
    bolt_r = p.motor_bolt_dia / 2
    head_r = p.motor_bolt_head_dia / 2
    cs = head_r - bolt_r
    with BuildPart() as h:
        with Locations((0, 0, -p.flange_t)):
            Cylinder(radius=case_r, height=p.flange_t + z_top, align=_MIN)
        # back carrier-bearing SEAT: a counterbore (presses the 30x37 OD) with a
        # shoulder ledge below it that the outer race lands on.
        with Locations((0, 0, p.z_bb)):
            Cylinder(radius=p.carrier_bearing_od / 2 + p.press_clear,
                     height=p.carrier_bearing_w, align=_MIN, mode=Mode.SUBTRACT)
        # below the seat: clear the rotating carrier hub and open the motor-shaft path
        # up to the eccentric, leaving the seat ledge at z_bb (hub OD < ledge < bearing OD)
        with Locations((0, 0, 0)):
            Cylinder(radius=p.carrier_bearing_id / 2 + p.run_clear, height=p.z_bb,
                     align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, p.z_bb + p.carrier_bearing_w)):
            Cylinder(radius=p.cyclo_cavity_r, height=z_top + 0.1, align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, p.z_disc - 0.5)):
            with PolarLocations(p.pin_R, p.n_pins):
                Cylinder(radius=p.pin_dia / 2, height=p.disc_thickness + 1.0,
                         align=_MIN, mode=Mode.SUBTRACT)
        for x, y in p.mount_holes():
            with Locations((x, y, -p.flange_t)):
                Cylinder(radius=bolt_r, height=p.flange_t, align=_MIN, mode=Mode.SUBTRACT)
            with Locations((x, y, -cs)):
                Cone(bottom_radius=bolt_r, top_radius=head_r, height=cs + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, -p.flange_t)):
            Cylinder(radius=p.motor_shaft_dia / 2 + 1.0, height=p.flange_t,
                     align=_MIN, mode=Mode.SUBTRACT)
        # Ø22 NEMA pilot register recess on the flange underside (centres the motor)
        if p.nema_pilot_depth > 0:
            with Locations((0, 0, -p.flange_t)):
                Cylinder(radius=p.nema_pilot_dia / 2 + p.run_clear, height=p.nema_pilot_depth,
                         align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, z_top - 5.0)):
            with PolarLocations(p.cap_bolt_r, p.n_case_bolts):
                Cylinder(radius=bolt_r - 0.4, height=5.0, align=_MIN, mode=Mode.SUBTRACT)
    return h.part


def make_output_cap(p: Params):
    """Front cap: holds the front carrier bearing, lets the output flange exit, bolts on."""
    case_r = p.case_od / 2
    base_z = p.z_split
    h_total = p.z_cap_top - base_z
    with BuildPart() as cap:
        with Locations((0, 0, base_z)):
            Cylinder(radius=case_r, height=h_total, align=_MIN)
        with Locations((0, 0, p.z_fb)):
            Cylinder(radius=p.carrier_bearing_od / 2 + p.press_clear,
                     height=p.carrier_bearing_w, align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, base_z)):
            Cylinder(radius=p.cyclo_cavity_r, height=p.z_fb - base_z + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, p.z_fb + p.carrier_bearing_w)):
            Cylinder(radius=p.out_flange_dia / 2 + p.run_clear, height=h_total,
                     align=_MIN, mode=Mode.SUBTRACT)
        with Locations((0, 0, base_z)):
            with PolarLocations(p.cap_bolt_r, p.n_case_bolts):
                Cylinder(radius=p.motor_bolt_dia / 2, height=6.0, align=_MIN, mode=Mode.SUBTRACT)
    return cap.part


def make_bearing(od: float, idia: float, w: float):
    """Simplified bearing body: an annular ring (OD x bore x width) standing in for
    the real ball bearing — enough to place + interference-check in the assembly."""
    outer = Cylinder(radius=od / 2, height=w, align=_MIN)
    bore = Pos(0, 0, -0.1) * Cylinder(radius=idia / 2, height=w + 0.2, align=_MIN)
    return outer - bore


def bearing_placements(p: Params):
    """(name, od, id, w, pos) for every bearing, on the assembly datum.
    The two shaft bearings nest concentrically inside the two carrier bearings;
    the 6700 rides the offset cam (so it sits at +E)."""
    cb = (p.carrier_bearing_od, p.carrier_bearing_id, p.carrier_bearing_w)
    sb = (p.shaft_bearing_od, p.shaft_bearing_id, p.shaft_bearing_w)
    ec = (p.ecc_bearing_od, p.ecc_bearing_id, p.ecc_bearing_w)
    return [
        ("brg_carrier_back",  *cb, (0, 0, p.z_bb)),
        ("brg_carrier_front", *cb, (0, 0, p.z_fb)),
        ("brg_shaft_back",    *sb, (0, 0, p.z_bb)),
        ("brg_shaft_front",   *sb, (0, 0, p.z_fb)),
        ("brg_disc_cam",      *ec, (p.eccentricity, 0, p.z_cam)),
    ]


# --------------------------------------------------------------------------- #
# ASSEMBLY + EXPORT
# --------------------------------------------------------------------------- #

def build(p: Params, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "housing": make_housing(p),
        "eccentric": make_eccentric(p),
        "disc": make_disc(p),
        "carrier_back": make_carrier_back(p),
        "carrier_front": make_carrier_front(p),
        "output_cap": make_output_cap(p),
    }
    for name, part in parts.items():
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name}.step / .stl")

    E = p.eccentricity
    placed = [
        ("housing", Pos(0, 0, 0) * parts["housing"]),
        ("eccentric", Pos(0, 0, p.z_bb) * parts["eccentric"]),
        ("disc", Pos(E, 0, p.z_disc) * parts["disc"]),
        ("carrier_back", Pos(0, 0, p.z_cback) * parts["carrier_back"]),
        ("carrier_front", Pos(0, 0, p.z_cfront) * parts["carrier_front"]),
        ("output_cap", Pos(0, 0, 0) * parts["output_cap"]),
    ]
    bodies = []
    for label, body in placed:
        body.label = label
        bodies.append(body)
    # printed-only assembly (for printing / BOM)
    export_step(Compound(children=list(bodies)), str(outdir / "assembly.step"))
    print("  wrote assembly.step  (6 printed parts)")

    # full assembly: printed parts + every bearing, each a SEPARATE labelled solid
    for name, od, idia, w, pos in bearing_placements(p):
        body = Pos(*pos) * make_bearing(od, idia, w)
        body.label = name
        bodies.append(body)
    export_step(Compound(children=bodies), str(outdir / "assembly_full.step"))
    print(f"  wrote assembly_full.step  ({len(bodies)} bodies: 6 printed + "
          f"{len(bodies)-6} bearings)")


def report(p: Params):
    print(f"=== {p.ratio:.0f}:1 straddle-carrier cycloidal — NEMA 17 input ===")
    print(f"input ............. NEMA 17 mount (Ø{p.motor_shaft_dia:.0f} shaft, "
          f"{p.nema_bolt_spacing:.0f}mm M3 square, Ø{p.nema_pilot_dia:.0f} register)")
    print(f"reduction ......... {p.ratio:.0f}:1 (output reversed)")
    print(f"carrier ........... 2-plate cage bolted by {p.n_out} output pins "
          f"(Ø{p.out_sleeve_od} roller + M{p.out_screw_dia} screw)")
    print(f"bearings .......... 2x carrier 30x37x4 (both ends) + 2x 7x13x4 input shaft "
          f"(nested) + 1x 6700 disc")
    print(f"output ............ Ø{p.out_flange_dia:.0f} flange, {p.out_bolt_circle:.0f}mm "
          f"bolt circle — carries overhung/moment load\n")


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
