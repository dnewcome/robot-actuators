"""
INVERTED single-stage planetary reducer — carrier (shaft) output, NEMA-17 in & out.

Re-derived from the SweepDynamics "4:1 Micro Planetary" vendor kit (12/18/48 gears,
Ø42 envelope, 7x13 + 30x37 bearing kit, Ø5 alu-roller planet axles). That product is
a BODY-output drive: sun in, carrier FIXED, the toothed housing spins as the output
(ratio = n_ring/n_sun = 48/12 = 4:1).

This design INVERTS it: hold the ring in a fixed housing and make the CARRIER the
output — a rotating shaft instead of a rotating body. With the same vendor gears the
ratio gains the +1 of a carrier output:

        ratio = 1 + n_ring/n_sun = 1 + 48/12 = 5:1   (carrier runs the SAME sense as
                                                      the sun; the body-out version reverses)

  INPUT  : NEMA-17 stepper bolts to the housing flange; its Ø5 shaft PRESS-FITS into
           the sun (no setscrew). The sun's Ø7 journal rides a 7x13x4 in the flange.
  RING   : 48 teeth, built into the fixed housing tube (carved from the wall).
  CARRIER: two-plate cage straddling 3 planets on Ø5 alu rollers (M3x5x10, kit part);
           the OUTPUT-side plate is a Ø30 hub riding a 30x37x4 in the end cap and
           carries the Ø5 output shaft. The big 30x37 gives the output real overhung
           capacity (the vendor used it for the whole rotating body).
  OUTPUT : the end cap presents a NEMA-17 face (31 mm M3 pattern + Ø22 pilot boss +
           Ø5 shaft) so a SECOND identical stage drops straight on — its flange bolts
           to this cap and its sun presses onto this output shaft. NEMA-17 in, out.

Parts (all printable + the vendor hardware kit): housing (stepper flange + RING +
input bearing pocket), sun, planet x3, carrier_bottom, carrier_top (hub+shaft),
cap (NEMA-17 output face + 30x37 pocket).
Hardware: 1x 7x13x4, 1x 30x37x4, 3x Ø5x10 alu rollers (M3 bore), 3x M3x20 button
head (cage), 4x M3 (stepper mount), 4x M2.5 (cap-to-housing), 4x M3 (output face).

Run with the repo venv:  .venv/bin/python planetary-inverted/drive.py
"""

import sys
from dataclasses import dataclass
from math import pi, sin, cos, sqrt
from pathlib import Path

# reuse the involute gear generator shared with the cycloidal package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycloidal"))
from gears import spur_gear, ring_gear  # noqa: E402

from build123d import (  # noqa: E402
    BuildPart, Box, Cone, Cylinder, Compound, Locations, PolarLocations, Pos,
    Mode, Align, export_step, export_stl,
)

_MIN = (Align.CENTER, Align.CENTER, Align.MIN)


@dataclass
class InvParams:
    # --- Reduction: vendor gearset, carrier OUT (ring fixed) -> 5:1 -------------
    # n_ring = n_sun + 2*n_planet = 12 + 36 = 48.  ratio = 1 + 48/12 = 5.0.
    n_sun: int = 12
    n_planet: int = 18
    n_planets: int = 3              # vendor uses 3; (n_sun+n_ring)%n_planets==0 -> 60%3=0
    gear_module: float = 0.677      # backed out of the vendor STEP (tip & ring-tip dia)
    gear_thickness: float = 9.5     # face width; the Ø5x10 roller sets a ~0.5 mm taller gap
    stage_eta: float = 0.94         # single printed planetary stage

    # --- Planet axles: Ø5 alu rollers (M3x5x10 kit part), clamp the 2-plate cage --
    roller_dia: float = 5.0         # roller OD the planet rotates on
    roller_len: float = 10.0        # sets the plate gap (gear_thickness + axial play)
    planet_bore_clear: float = 0.20 # planet bore radial clearance over the roller
    cage_screw_dia: float = 3.0     # M3 down the roller, threads into the top hub
    cage_screw_pilot: float = 2.5   # tap / heat-set pilot in the top hub
    cage_screw_head_dia: float = 6.0  # M3 button-head OD (bottom-plate counterbore)
    cage_screw_head_h: float = 2.4
    n_cage_posts: int = 3           # standoff posts BETWEEN the planets (own screws+spacers)
    ring_rim: float = 2.5           # ring body outside the root circle

    # --- Sun: PRESS-FIT onto the Ø5 NEMA-17 stepper shaft (no setscrew) ---------
    motor_shaft_dia: float = 5.0    # NEMA-17 stepper shaft
    shaft_press_fit: float = 0.05   # radial interference: bore = shaft - 2*this
    sun_journal_dia: float = 7.0    # journal below the gear, rides the 7x13 bore
    sun_journal_h: float = 5.0      # journal length engaged below the gear plane

    # --- Bearings (vendor kit) -------------------------------------------------
    in_bearing_id: float = 7.0      # 7x13x4 on the sun journal
    in_bearing_od: float = 13.0
    in_bearing_w: float = 4.0
    out_bearing_id: float = 30.0    # 30x37x4 thin-section on the carrier output hub
    out_bearing_od: float = 37.0
    out_bearing_w: float = 4.0

    # --- Input flange: NEMA-17 stepper mount -----------------------------------
    nema_bolt_spacing: float = 31.0     # 31 mm square M3 pattern
    nema_bolt_dia: float = 3.4          # M3 clearance (stepper mount)
    nema_bolt_head_dia: float = 6.0     # flat-head OD (countersink mouth)
    nema_pilot_dia: float = 22.0        # register boss Ø22
    in_pilot_depth: float = 2.0         # recess on the flange underside (accepts stepper boss)

    # --- Output: carrier shaft + NEMA-17 face on the cap -----------------------
    out_shaft_dia: float = 5.0          # Ø5 output (a second stage presses its sun on)
    out_shaft_protrude: float = 15.0    # length past the cap face
    out_shaft_flat_depth: float = 0.0   # >0 for a D-flat
    out_pilot_boss: float = 2.0         # Ø22 raised boss on the output face (0 = none)
    out_nema_pilot_dia: float = 22.0
    out_bolt_pilot: float = 2.5         # M3 tap pilot Ø in the output face
    out_bolt_depth: float = 5.0         # blind tapped depth

    # --- Shell / fits ----------------------------------------------------------
    case_wall: float = 1.5          # radial wall outside the ring (-> Ø42-ish like vendor)
    flange_t: float = 8.0           # stepper-mount flange (pockets the 7x13 + pilot)
    cap_t: float = 3.0              # cap roof above the 30x37
    carrier_bot_t: float = 3.0      # input-side carrier plate
    carrier_top_t: float = 3.0      # output-side plate (below the hub)
    motor_face_gap: float = 0.5     # flange top -> bottom plate
    gear_carrier_gap: float = 0.5   # gear top -> top hub underside (axial play in the gap)
    press_clear: float = 0.04
    run_clear: float = 0.15
    case_bolt_dia: float = 2.5      # M2.5 cap-to-housing
    case_bolt_head_dia: float = 4.5
    n_case_bolts: int = 4

    output_lash_deg: float = 0.25

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
        return (self.sun_pcd + self.planet_pcd) / 2.0

    @property
    def planet_tip_dia(self) -> float:
        return self.planet_pcd + 2 * self.gear_module

    @property
    def sun_tip_dia(self) -> float:
        return self.sun_pcd + 2 * self.gear_module

    @property
    def ring_inner_tip_r(self) -> float:
        """Inner radius of the ring teeth (carrier rotates inside this)."""
        return self.ring_pcd / 2.0 - self.gear_module

    @property
    def ring_root_r(self) -> float:
        return self.ring_pcd / 2.0 + 1.25 * self.gear_module

    @property
    def ring_od(self) -> float:
        return 2 * self.ring_root_r + 2 * self.ring_rim

    @property
    def case_od(self) -> float:
        return self.ring_od + 2 * self.case_wall

    @property
    def planet_bore_dia(self) -> float:
        return self.roller_dia + 2 * self.planet_bore_clear

    @property
    def carrier_bot_od(self) -> float:
        """Bottom plate Ø: past the planet rollers, clear of the ring teeth."""
        return 2 * (self.carrier_radius + self.roller_dia / 2 + 1.0)

    @property
    def carrier_top_od(self) -> float:
        """Top hub Ø = the 30x37 bore (it IS the output journal)."""
        return self.out_bearing_id

    @property
    def hub_clear_bore_r(self) -> float:
        """Bottom-plate centre bore: wide enough the cage slides over the sun gear."""
        return self.sun_tip_dia / 2 + 0.5

    @property
    def case_bolt_r(self) -> float:
        """Cap/housing bolt circle: just outside the 30x37 pocket."""
        return self.out_bearing_od / 2 + self.case_bolt_dia / 2 + 0.3

    @property
    def out_plate(self) -> float:
        """Square NEMA-17 end-plate side (holds the 31 mm pattern + case bolts)."""
        nema = self.nema_bolt_spacing / 2 + self.nema_bolt_head_dia / 2 + 1.0
        bolt = self.case_bolt_r + self.case_bolt_head_dia / 2 + 1.0
        return 2 * max(nema, bolt)

    # axial datum: z=0 at the stepper mounting face (flange underside)
    @property
    def in_bearing_floor_z(self) -> float:
        """7x13 seats here, on a ledge above the pilot recess."""
        return self.flange_t - self.in_bearing_w

    @property
    def carrier_bot_z(self) -> float:
        return self.flange_t + self.motor_face_gap

    @property
    def gear_z(self) -> float:
        """Gear plane = top of the bottom plate."""
        return self.carrier_bot_z + self.carrier_bot_t

    @property
    def carrier_top_z(self) -> float:
        """Top hub underside seats on the roller tops (rollers set the gap)."""
        return self.gear_z + self.roller_len

    @property
    def housing_h(self) -> float:
        """Tube top = top-hub plate top (the cap sits here)."""
        return self.carrier_top_z + self.carrier_top_t

    @property
    def out_bearing_floor_z(self) -> float:
        """30x37 seats in the cap just above the tube top."""
        return self.housing_h

    @property
    def sun_z(self) -> float:
        """Sun base (journal bottom) sits in the flange bearing."""
        return self.in_bearing_floor_z

    @property
    def carrier_bore_below_r(self) -> float:
        return self.carrier_bot_od / 2 + self.run_clear

    @property
    def carrier_bore_above_r(self) -> float:
        return self.carrier_top_od / 2 + self.run_clear

    @property
    def post_angle_start(self) -> float:
        """Standoff posts sit midway between planets (half the planet pitch, degrees)."""
        return 360.0 / self.n_planets / 2.0

    def nema_holes(self):
        h = self.nema_bolt_spacing / 2
        return [(h, h), (-h, h), (-h, -h), (h, -h)]


# --------------------------------------------------------------------------- #
# FEASIBILITY VALIDATOR
# --------------------------------------------------------------------------- #

def validate(p: InvParams) -> bool:
    ok = True
    print(f"\n=== Inverted planetary feasibility  ({p.ratio:.2f}:1 carrier-out, {p.n_planets} planets) ===")
    print(f"teeth  sun {p.n_sun} / planet {p.n_planet} / ring {p.n_ring}   module {p.gear_module}")
    print(f"PCD    sun Ø{p.sun_pcd:.2f} / planet Ø{p.planet_pcd:.2f} / ring Ø{p.ring_pcd:.2f}")
    print(f"case   Ø{p.case_od:.1f} tube, {p.out_plate:.1f} sq plates, {p.housing_h:.1f} mm body")

    def check(label, cond, detail):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")

    # geometry closes
    check("ring = sun + 2*planet", p.n_ring == p.n_sun + 2 * p.n_planet,
          f"{p.n_ring} == {p.n_sun}+2*{p.n_planet}")
    check("equal planet spacing", (p.n_sun + p.n_ring) % p.n_planets == 0,
          f"({p.n_sun}+{p.n_ring}) % {p.n_planets} = {(p.n_sun + p.n_ring) % p.n_planets}")

    chord = 2 * p.carrier_radius * sin(pi / p.n_planets)
    gap = chord - p.planet_tip_dia
    check("planets clear each other", gap >= 0.8, f"{gap:.2f} mm tip-to-tip gap")

    # sun press fit onto the Ø5 stepper shaft
    bore_wall = p.sun_journal_dia / 2 - (p.motor_shaft_dia / 2 - p.shaft_press_fit)
    check("sun journal wall over shaft", bore_wall >= 0.9,
          f"{bore_wall:.2f} mm around Ø{p.motor_shaft_dia} shaft (press {p.shaft_press_fit*1000:.0f} µm)")
    check("sun journal rides 7x13 bore", abs(p.sun_journal_dia - p.in_bearing_id) < 0.01,
          f"journal Ø{p.sun_journal_dia} == bearing bore Ø{p.in_bearing_id}")

    # input bearing seats in the flange above the pilot recess
    check("7x13 seats on flange ledge", p.in_bearing_w + p.in_pilot_depth + 0.5 <= p.flange_t,
          f"bearing {p.in_bearing_w} + pilot {p.in_pilot_depth} + 0.5 <= flange {p.flange_t}")

    # ring fits the case wall; carrier rotates inside the ring teeth
    cw = (p.case_od - p.ring_od) / 2
    check("case wall around ring", cw >= 1.4, f"{cw:.2f} mm")
    cclear = p.ring_inner_tip_r - (p.carrier_bot_od / 2 + p.run_clear)
    check("bottom plate clears ring teeth", cclear >= 0.4, f"{cclear:.2f} mm")

    # planet on roller, roller sets the plate gap
    planet_root_r = p.planet_pcd / 2 - 1.25 * p.gear_module
    rim = planet_root_r - p.planet_bore_dia / 2
    check("planet rim over roller", rim >= 1.0,
          f"{rim:.2f} mm (root R {planet_root_r:.2f} - bore R {p.planet_bore_dia/2:.2f})")
    axial = p.roller_len - p.gear_thickness
    check("roller sets axial play", 0.2 <= axial <= 1.0, f"{axial:.2f} mm gap-vs-gear")
    # standoff posts (between planets) must clear the adjacent planet tips
    d_post = 2 * p.carrier_radius * sin((pi * p.post_angle_start / 180.0) / 2.0)
    post_clear = d_post - p.planet_tip_dia / 2 - p.roller_dia / 2
    check("standoff posts clear planets", post_clear >= 0.5,
          f"{post_clear:.2f} mm (post Ø{p.roller_dia} at the {p.post_angle_start:.0f}° gaps)")

    # OUTPUT hub + 30x37 + cap
    check("30x37 OD fits the tube wall", p.out_bearing_od + 2 * 1.5 <= p.case_od + 0.6,
          f"bearing Ø{p.out_bearing_od} + 3 <= case Ø{p.case_od:.1f}")
    check("output hub == 30x37 bore", abs(p.carrier_top_od - p.out_bearing_id) < 0.01,
          f"hub Ø{p.carrier_top_od} rides bore Ø{p.out_bearing_id}")
    check("cap bolts clear the 30x37 pocket",
          p.case_bolt_r - p.case_bolt_dia / 2 >= p.out_bearing_od / 2 + 0.0,
          f"bolt inner R {p.case_bolt_r - p.case_bolt_dia/2:.2f} vs pocket R {p.out_bearing_od/2:.1f}")
    check("cap bolts land on the square plate",
          p.case_bolt_r + p.case_bolt_head_dia / 2 + 0.5 <= p.out_plate / 2,
          f"bolt edge {p.case_bolt_r + p.case_bolt_head_dia/2:.1f} vs plate half {p.out_plate/2:.1f}")

    # NEMA-17 patterns (in flange + out cap) land on the square plates
    nr = p.nema_bolt_spacing / 2
    nb_edge = nr + p.nema_bolt_head_dia / 2 + 0.5
    check("NEMA holes on the plate", nb_edge <= p.out_plate / 2,
          f"hole+head {nb_edge:.1f} vs plate half {p.out_plate/2:.1f}")
    # min centre-to-centre between any NEMA hole (diagonals) and any case bolt (axes)
    case_xy = [(p.case_bolt_r * cos(2 * pi * k / p.n_case_bolts),
                p.case_bolt_r * sin(2 * pi * k / p.n_case_bolts)) for k in range(p.n_case_bolts)]
    dmin = min(sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
               for hx, hy in p.nema_holes() for cx, cy in case_xy)
    need = p.nema_bolt_head_dia / 2 + p.case_bolt_head_dia / 2 + 0.5
    check("NEMA holes clear the case bolts", dmin >= need,
          f"{dmin:.1f} mm centre gap >= {need:.1f} needed")

    print("  ----")
    print(f"  RESULT: {'ALL GOOD' if ok else 'CONSTRAINTS VIOLATED'}")
    return ok


# --------------------------------------------------------------------------- #
# PART BUILDERS  (each in a local frame, base at z=0 for printing)
# --------------------------------------------------------------------------- #

def make_housing(p: InvParams):
    """Fixed housing: square NEMA-17 stepper flange (z0..flange_t) + round Ø42 tube with
    the 48-tooth RING built into the wall. The 7x13 pockets low in the flange on a ledge;
    the stepper bolts up through countersunk corners; the carrier spins in the bore."""
    case_r = p.case_od / 2
    bolt_r = p.nema_bolt_dia / 2
    head_r = p.nema_bolt_head_dia / 2
    cs = head_r - bolt_r                          # 90° countersink depth
    ft = p.flange_t
    gz, gt = p.gear_z, p.gear_thickness
    floor = p.in_bearing_floor_z
    with BuildPart() as h:
        # square stepper flange, then the round tube above it
        Box(p.out_plate, p.out_plate, ft, align=_MIN)
        with Locations((0, 0, ft)):
            Cylinder(radius=case_r, height=p.housing_h - ft, align=_MIN)
        # ---- flange interior bores (bottom -> top) ----
        # Ø22 pilot recess accepts the stepper register boss
        Cylinder(radius=p.nema_pilot_dia / 2 + p.run_clear, height=p.in_pilot_depth,
                 align=_MIN, mode=Mode.SUBTRACT)
        # journal clearance between pilot recess and the bearing ledge
        with Locations((0, 0, p.in_pilot_depth)):
            Cylinder(radius=p.sun_journal_dia / 2 + p.run_clear,
                     height=floor - p.in_pilot_depth + 0.01, align=_MIN, mode=Mode.SUBTRACT)
        # 7x13 bearing pocket (seats on the ledge at `floor`)
        with Locations((0, 0, floor)):
            Cylinder(radius=p.in_bearing_od / 2 + p.press_clear, height=p.in_bearing_w + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        # below-gear cavity: bottom carrier plate spins here
        with Locations((0, 0, ft)):
            Cylinder(radius=p.carrier_bore_below_r, height=gz - ft, align=_MIN, mode=Mode.SUBTRACT)
        # above-gear cavity: top hub spins here, up to the tube top
        with Locations((0, 0, gz + gt)):
            Cylinder(radius=p.carrier_bore_above_r, height=p.housing_h - (gz + gt) + 0.1,
                     align=_MIN, mode=Mode.SUBTRACT)
        # stepper mount: 4 countersunk M3 through the flange corners (heads open up at z=ft)
        for x, y in p.nema_holes():
            with Locations((x, y, 0)):
                Cylinder(radius=bolt_r, height=ft, align=_MIN, mode=Mode.SUBTRACT)
            with Locations((x, y, ft - cs)):
                Cone(bottom_radius=bolt_r, top_radius=head_r, height=cs + 0.01,
                     align=_MIN, mode=Mode.SUBTRACT)
        # cap-to-housing bolt pilots, down into the tube top wall (on axis)
        with Locations((0, 0, p.housing_h - 6.0)):
            with PolarLocations(p.case_bolt_r, p.n_case_bolts):
                Cylinder(radius=p.case_bolt_dia / 2 - 0.4, height=6.0, align=_MIN, mode=Mode.SUBTRACT)
    body = h.part
    # carve the internal ring teeth at the gear plane (teeth remain integral to the wall)
    ring_neg = (Pos(0, 0, gt / 2) * Cylinder(radius=p.ring_root_r, height=gt)
                - ring_gear(p.gear_module, p.n_ring, gt, rim=p.ring_rim))
    return body - Pos(0, 0, p.gear_z) * ring_neg


def make_sun(p: InvParams):
    """Sun gear + Ø7 journal, bored as an interference PRESS FIT onto the Ø5 stepper
    shaft. Journal rides the 7x13; gear meshes the planets."""
    bore_r = p.motor_shaft_dia / 2.0 - p.shaft_press_fit
    gear = Pos(0, 0, p.sun_journal_h) * spur_gear(p.gear_module, p.n_sun, p.gear_thickness, bore=0.0)
    journal = Pos(0, 0, p.sun_journal_h / 2) * Cylinder(radius=p.sun_journal_dia / 2, height=p.sun_journal_h)
    sun = gear + journal
    H = p.sun_journal_h + p.gear_thickness
    bore = Cylinder(radius=bore_r, height=3 * H)
    if p.out_shaft_flat_depth > 0:  # (unused for the sun; round press bore)
        pass
    return sun - bore


def make_planet(p: InvParams):
    """Planet gear bored with running clearance over its Ø5 roller axle."""
    return spur_gear(p.gear_module, p.n_planet, p.gear_thickness, bore=p.planet_bore_dia)


def make_carrier_bottom(p: InvParams):
    """Input-side carrier plate: wide central bore (slides over the pressed sun gear),
    plus a cage screw at EVERY station — 3 down the planet rollers AND 3 standoff posts
    in the gaps between planets — each with a button-head counterbore on the underside.
    The posts (own Ø5x10 spacers) bolt the cage rigidly without touching the planets."""
    stations = [(p.n_planets, 0.0), (p.n_cage_posts, p.post_angle_start)]  # (count, start°)
    with BuildPart() as c:
        Cylinder(radius=p.carrier_bot_od / 2, height=p.carrier_bot_t, align=_MIN)
        # central clearance bore for the sun gear (assembly)
        Cylinder(radius=p.hub_clear_bore_r, height=p.carrier_bot_t, align=_MIN, mode=Mode.SUBTRACT)
        for count, start in stations:
            with PolarLocations(p.carrier_radius, count, start_angle=start):
                Cylinder(radius=p.cage_screw_dia / 2 + 0.25, height=p.carrier_bot_t,
                         align=_MIN, mode=Mode.SUBTRACT)
            with PolarLocations(p.carrier_radius, count, start_angle=start):
                Cylinder(radius=p.cage_screw_head_dia / 2, height=p.cage_screw_head_h,
                         align=_MIN, mode=Mode.SUBTRACT)
    return c.part


def make_carrier_top(p: InvParams):
    """Output-side carrier: a Ø30 hub that rides the 30x37 and carries the Ø5 output
    shaft. The roller tops seat against its underside; the cage screws thread into it."""
    sd = p.out_shaft_dia
    hub_h = p.carrier_top_t + p.out_bearing_w + 0.5      # plate-in-tube + bearing journal
    shaft_len = p.cap_t + p.out_shaft_protrude + 0.5
    with BuildPart() as c:
        Cylinder(radius=p.carrier_top_od / 2, height=hub_h, align=_MIN)
        with Locations((0, 0, hub_h)):
            Cylinder(radius=sd / 2, height=shaft_len, align=_MIN)
        # cage-screw pilots at every station (planet rollers + standoff posts)
        for count, start in [(p.n_planets, 0.0), (p.n_cage_posts, p.post_angle_start)]:
            with PolarLocations(p.carrier_radius, count, start_angle=start):
                Cylinder(radius=p.cage_screw_pilot / 2, height=hub_h - 0.5,
                         align=_MIN, mode=Mode.SUBTRACT)
    shaft = c.part
    if p.out_shaft_flat_depth > 0:
        flat = sd / 2 - p.out_shaft_flat_depth
        z0 = hub_h + p.cap_t + 0.5
        shaft = shaft - Pos(flat + sd, 0, z0 + p.out_shaft_protrude / 2) * Box(2 * sd, 4 * sd, p.out_shaft_protrude + 0.2)
    return shaft


def make_cap(p: InvParams):
    """End cap = NEMA-17 OUTPUT face: square plate + 30x37 pocket + Ø22 pilot boss +
    Ø5 shaft hole + 31 mm M3 tapped pattern, so a second identical stage bolts on."""
    plate = p.out_plate
    th = p.out_bearing_w + p.cap_t
    bz = th + p.out_pilot_boss                  # total incl. raised pilot boss
    with BuildPart() as cap:
        Box(plate, plate, th, align=_MIN)
        # raised Ø22 register boss on the output face (top)
        if p.out_pilot_boss > 0:
            with Locations((0, 0, th)):
                Cylinder(radius=p.out_nema_pilot_dia / 2, height=p.out_pilot_boss, align=_MIN)
        # 30x37 pocket on the carrier-facing side (z=0)
        Cylinder(radius=p.out_bearing_od / 2 + p.press_clear, height=p.out_bearing_w,
                 align=_MIN, mode=Mode.SUBTRACT)
        # Ø5 output shaft through-hole (incl. through the pilot boss)
        Cylinder(radius=p.out_shaft_dia / 2 + p.run_clear, height=bz, align=_MIN, mode=Mode.SUBTRACT)
        # NEMA-17 output mounting holes: blind M3, tapped from the output face (top)
        for x, y in p.nema_holes():
            with Locations((x, y, bz - p.out_bolt_depth)):
                Cylinder(radius=p.out_bolt_pilot / 2, height=p.out_bolt_depth + 0.01,
                         align=_MIN, mode=Mode.SUBTRACT)
        # cap-to-housing screws (clearance, on axis -> miss the NEMA diagonals)
        with PolarLocations(p.case_bolt_r, p.n_case_bolts):
            Cylinder(radius=p.case_bolt_dia / 2, height=th, align=_MIN, mode=Mode.SUBTRACT)
    return cap.part


# --------------------------------------------------------------------------- #
# BEARINGS (simple annular envelopes, for the full STEP) + ASSEMBLY/EXPORT
# --------------------------------------------------------------------------- #

def make_bearing(od, idia, w):
    outer = Cylinder(radius=od / 2, height=w, align=_MIN)
    bore = Pos(0, 0, -0.1) * Cylinder(radius=idia / 2, height=w + 0.2, align=_MIN)
    return outer - bore


def bearing_placements(p: InvParams):
    return [
        ("brg_input_7x13", p.in_bearing_od, p.in_bearing_id, p.in_bearing_w,
         (0, 0, p.in_bearing_floor_z)),
        ("brg_output_30x37", p.out_bearing_od, p.out_bearing_id, p.out_bearing_w,
         (0, 0, p.out_bearing_floor_z)),
    ]


def build(p: InvParams, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    parts = {
        "housing": make_housing(p),
        "sun": make_sun(p),
        "planet": make_planet(p),
        "carrier_bottom": make_carrier_bottom(p),
        "carrier_top": make_carrier_top(p),
        "cap": make_cap(p),
    }
    for name, part in parts.items():
        export_step(part, str(outdir / f"{name}.step"))
        export_stl(part, str(outdir / f"{name}.stl"))
        print(f"  wrote {name}.step / .stl   (valid={part.is_valid})")

    # assemble on the shared axial datum (z=0 at the stepper face)
    asm = [Pos(0, 0, 0) * parts["housing"]]
    asm.append(Pos(0, 0, p.sun_z) * parts["sun"])
    asm.append(Pos(0, 0, p.carrier_bot_z) * parts["carrier_bottom"])
    for i in range(p.n_planets):
        ang = 2 * pi * i / p.n_planets
        x, y = p.carrier_radius * cos(ang), p.carrier_radius * sin(ang)
        asm.append(Pos(x, y, p.gear_z) * parts["planet"])
    asm.append(Pos(0, 0, p.carrier_top_z) * parts["carrier_top"])
    asm.append(Pos(0, 0, p.housing_h) * parts["cap"])
    # Ø5x10 rollers: planet axles + standoff posts (kit hardware, shown for reference)
    roller = Cylinder(radius=p.roller_dia / 2, height=p.roller_len, align=_MIN)
    for count, start in [(p.n_planets, 0.0), (p.n_cage_posts, p.post_angle_start)]:
        for i in range(count):
            ang = start * pi / 180 + 2 * pi * i / count
            x, y = p.carrier_radius * cos(ang), p.carrier_radius * sin(ang)
            asm.append(Pos(x, y, p.gear_z) * roller)
    export_step(Compound(children=[a for a in asm]), str(outdir / "assembly.step"))
    print("  wrote assembly.step")

    # full assembly incl. bearings as separate labelled bodies
    full = list(asm)
    for name, od, idia, w, pos in bearing_placements(p):
        b = Pos(*pos) * make_bearing(od, idia, w)
        b.label = name
        full.append(b)
    for i, body in enumerate(full):
        if not getattr(body, "label", ""):
            body.label = list(parts.keys())[i] if i < len(parts) else f"body{i}"
    export_step(Compound(children=full), str(outdir / "assembly_full.step"))
    print("  wrote assembly_full.step  (printed parts + 2 bearings)")


def report(p: InvParams):
    """Torque / speed for a typical NEMA-17 stepper driving this single inverted stage."""
    hold = 0.40                                  # N·m, a common NEMA-17 holding torque
    out_torque = hold * p.ratio * p.stage_eta
    print(f"\n=== NEMA-17 stepper + {p.ratio:.0f}:1 inverted planetary (carrier out) ===")
    print(f"reduction ......... {p.ratio:.0f}:1  (vendor 12/18/48 gears, carrier out, same sense as sun)")
    print(f"torque ............ ~{hold*1000:.0f} mN·m stepper -> ~{out_torque*1000:.0f} mN·m output "
          f"(η {p.stage_eta:.0%})")
    print(f"backlash .......... ~{p.output_lash_deg:.2f}° at output (single stage)")
    print(f"input ............. NEMA-17 mount ({p.out_plate:.0f}mm sq, 31mm M3) + Ø{p.motor_shaft_dia:.0f} "
          f"press-fit sun ({p.shaft_press_fit*1000:.0f} µm), no setscrew")
    print(f"output ............ NEMA-17 face + Ø{p.out_shaft_dia:.0f} shaft ({p.out_shaft_protrude:.0f}mm out) "
          f"on a 30x37 bearing -> chain a 2nd stage")
    print(f"carrier ........... 2-plate cage: {p.n_planets} planets on Ø{p.roller_dia:.0f}x{p.roller_len:.0f} rollers "
          f"+ {p.n_cage_posts} standoff posts between them, all M{p.cage_screw_dia:.0f} clamped")
    print(f"bearings .......... input 7x13x4 (sun) + output 30x37x4 (carrier hub)")
    print(f"envelope .......... Ø{p.case_od:.1f} tube, {p.housing_h:.1f}mm body (vendor-matched)")


if __name__ == "__main__":
    p = InvParams()
    ok = validate(p)
    report(p)
    if ok:
        out = Path(__file__).resolve().parent / "out"
        print(f"\nbuilding -> {out}")
        build(p, out)
    else:
        print("\nfix constraints before building.")
        sys.exit(1)
