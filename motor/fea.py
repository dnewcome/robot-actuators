"""
2-D magnetostatic FEA of a CUSTOM TORQUE MOTOR for the cycloidal robot joint.

This is the electromagnetic counterpart to drive.py (the mechanical CAD) and sim.py
(the MuJoCo dynamics): a parametric, code-driven finite-element model of the motor that
*produces* the torque the rest of the stack assumes. The job mirrors what a FEMM motor
study does, but native on Linux and in the same parametric idiom as the rest of the repo:

  geometry (parametric)  ->  gmsh mesh  ->  nonlinear magnetostatic solve (scikit-fem)
  ->  air-gap Maxwell-stress torque  ->  Kt / cogging / back-EMF  ->  feeds the sim model

Topology: an OUTRUNNER (rotor + magnets on the OUTSIDE, slotted stator inside) so the
rotor wraps the cycloidal and drives its eccentric "from the outside" — the big air-gap
radius is what buys the torque, the high pole count buys the holding/detent torque.

This module (Step 1) builds the parametric cross-section and meshes it. Run it to render
the coloured region map and confirm the geometry before any physics is added:

    ../.venv/bin/python motor/fea.py            # build mesh, write out/mesh.png
"""

import sys
from dataclasses import dataclass, field
from math import pi, cos, sin, atan2, hypot, tau
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

MU0 = 4e-7 * pi


@dataclass
class MotorParams:
    # --- Pole/slot count (fractional-slot concentrated winding) -----------------
    # 24 slots / 22 poles: a classic high-torque-density combo (q = 0.36), low
    # cogging, coil-per-tooth — the gimbal / robot-joint recipe.
    slots: int = 24
    poles: int = 22                      # pole_pairs = poles/2 = 11

    stack_len: float = 18.0              # axial length [mm] — sets torque linearly

    # --- Radial build, OUTRUNNER, from the shaft bore outward [mm] --------------
    # Bore is sized to clear the cycloidal eccentric / coupling (Ø5 shaft + hub).
    bore_r: float = 9.0                  # inner radius of the stator yoke (Ø18 bore)
    slot_bottom_r: float = 21.0          # where the slot opening / tooth body starts
    stator_or: float = 28.0              # stator outer radius = tooth-tip radius
    air_gap: float = 0.6                 # radial mechanical gap [mm]
    magnet_thick: float = 3.0            # magnet radial thickness [mm]
    rotor_back_iron: float = 3.0         # rotor back-iron radial thickness [mm]

    # --- Magnet + tooth arc fractions (of their pitch) --------------------------
    magnet_arc_frac: float = 0.83        # magnet pole-arc / pole-pitch
    slot_arc_frac: float = 0.50          # slot opening / slot-pitch (rest = tooth)

    # --- Materials --------------------------------------------------------------
    magnet_Br: float = 1.28              # NdFeB N42 remanence [T]
    magnet_mur: float = 1.05             # recoil permeability
    iron_mur_linear: float = 2000.0      # linear stand-in until the B-H curve is wired
    air_mur: float = 1.0
    copper_mur: float = 1.0

    # --- Winding / drive --------------------------------------------------------
    turns_per_coil: int = 60             # series turns per tooth coil
    slot_fill: float = 0.40              # copper packing fraction in the slot
    phase_current: float = 6.0           # peak phase current [A]

    # --- Mesh sizing [mm] -------------------------------------------------------
    mesh_gap: float = 0.25               # element size in the air gap (resolve flux)
    mesh_coarse: float = 1.6             # element size far from the gap

    # ---- derived ---------------------------------------------------------------
    @property
    def pole_pairs(self) -> int:
        return self.poles // 2

    @property
    def magnet_inner_r(self) -> float:
        return self.stator_or + self.air_gap

    @property
    def magnet_outer_r(self) -> float:
        return self.magnet_inner_r + self.magnet_thick

    @property
    def rotor_outer_r(self) -> float:
        return self.magnet_outer_r + self.rotor_back_iron

    @property
    def gap_mid_r(self) -> float:
        return 0.5 * (self.stator_or + self.magnet_inner_r)

    @property
    def ratio_note(self) -> str:
        return f"{self.slots}s/{self.poles}p (pp={self.pole_pairs})"


# ---------------------------------------------------------------------------
# Region classification — every mesh element is one of these, decided purely by
# its centroid (radius + angle), so a rotor rotation is just an angle shift.
# ---------------------------------------------------------------------------
SHAFT_AIR, STATOR_IRON, COIL, AIRGAP, MAGNET, MAGNET_AIR, ROTOR_IRON = range(7)
REGION_NAMES = {
    SHAFT_AIR: "shaft_air", STATOR_IRON: "stator_iron", COIL: "coil",
    AIRGAP: "airgap", MAGNET: "magnet", MAGNET_AIR: "magnet_air",
    ROTOR_IRON: "rotor_iron",
}
REGION_COLORS = {
    SHAFT_AIR: "#e8e8e8", STATOR_IRON: "#5b6470", COIL: "#c87a2f",
    AIRGAP: "#f4f4f4", MAGNET: "#c0392b", MAGNET_AIR: "#fbeae8",
    ROTOR_IRON: "#34404d",
}


def _in_arc(theta: float, centers: np.ndarray, half_width: float) -> bool:
    """Is angle theta within half_width of any of the given arc centers?"""
    d = np.abs((theta - centers + pi) % tau - pi)
    return bool(np.any(d <= half_width))


def classify(x: float, y: float, p: MotorParams, rotor_angle: float) -> int:
    """Map a centroid (mm) to a region id. rotor_angle in radians."""
    r = hypot(x, y)
    th = atan2(y, x)

    if r < p.bore_r:
        return SHAFT_AIR
    if r < p.stator_or:
        # slot pitch is fixed to the stator; slots centred on slot-pitch lines
        slot_pitch = tau / p.slots
        slot_centers = np.arange(p.slots) * slot_pitch
        if r > p.slot_bottom_r and _in_arc(th, slot_centers, 0.5 * p.slot_arc_frac * slot_pitch):
            return COIL
        return STATOR_IRON
    if r < p.magnet_inner_r:
        return AIRGAP
    if r < p.magnet_outer_r:
        pole_pitch = tau / p.poles
        mag_centers = np.arange(p.poles) * pole_pitch + rotor_angle
        if _in_arc(th, mag_centers, 0.5 * p.magnet_arc_frac * pole_pitch):
            return MAGNET
        return MAGNET_AIR
    return ROTOR_IRON


# ---------------------------------------------------------------------------
# Mesh — built in gmsh's OCC kernel as overlapping disks/annuli + wedge tools,
# fragmented into one conformal partition, then each face tagged by its centroid.
# Geometry is built in METRES (radii_mm * 1e-3) so the solver gets SI directly.
# ---------------------------------------------------------------------------
def build_mesh(p: MotorParams, rotor_angle: float = 0.0, verbose: bool = False):
    import gmsh

    mm = 1e-3
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("torque_motor")
        occ = gmsh.model.occ

        def annulus(r_in, r_out):
            o = occ.addDisk(0, 0, 0, r_out * mm, r_out * mm)
            if r_in <= 0:
                return o
            i = occ.addDisk(0, 0, 0, r_in * mm, r_in * mm)
            out, _ = occ.cut([(2, o)], [(2, i)])
            return out[0][1]

        def wedge(r_in, r_out, a_center, a_half):
            """Annular sector surface between radii and angles [a_center±a_half]."""
            cp = occ.addPoint(0, 0, 0)
            a0, a1 = a_center - a_half, a_center + a_half
            pin0 = occ.addPoint(r_in * mm * cos(a0), r_in * mm * sin(a0), 0)
            pout0 = occ.addPoint(r_out * mm * cos(a0), r_out * mm * sin(a0), 0)
            pin1 = occ.addPoint(r_in * mm * cos(a1), r_in * mm * sin(a1), 0)
            pout1 = occ.addPoint(r_out * mm * cos(a1), r_out * mm * sin(a1), 0)
            l_r0 = occ.addLine(pin0, pout0)
            arc_o = occ.addCircleArc(pout0, cp, pout1)
            l_r1 = occ.addLine(pout1, pin1)
            arc_i = occ.addCircleArc(pin1, cp, pin0)
            loop = occ.addCurveLoop([l_r0, arc_o, l_r1, arc_i])
            return occ.addPlaneSurface([loop])

        surfaces = []
        surfaces.append(annulus(0, p.bore_r))                       # shaft bore
        surfaces.append(annulus(p.bore_r, p.stator_or))             # stator blank
        slot_pitch = tau / p.slots
        for k in range(p.slots):                                    # slot/coil wedges
            surfaces.append(wedge(p.slot_bottom_r, p.stator_or,
                                  k * slot_pitch, 0.5 * p.slot_arc_frac * slot_pitch))
        surfaces.append(annulus(p.stator_or, p.magnet_inner_r))     # air gap
        surfaces.append(annulus(p.magnet_inner_r, p.magnet_outer_r))  # magnet ring blank
        pole_pitch = tau / p.poles
        for k in range(p.poles):                                    # magnet wedges
            surfaces.append(wedge(p.magnet_inner_r, p.magnet_outer_r,
                                  k * pole_pitch + rotor_angle,
                                  0.5 * p.magnet_arc_frac * pole_pitch))
        surfaces.append(annulus(p.magnet_outer_r, p.rotor_outer_r))  # back iron

        occ.synchronize()
        # Fragment everything into one conformal partition (shared boundaries glued,
        # so every triangle lands entirely inside exactly one region).
        all_dt = [(2, s) for s in surfaces]
        occ.fragment(all_dt, [])
        occ.synchronize()
        # A single physical group keeps gmsh from discarding faces at meshing.
        gmsh.model.addPhysicalGroup(2, [t for (_, t) in gmsh.model.getEntities(2)], name="dom")

        # Mesh sizing: fine in the gap, coarse elsewhere (MathEval field on radius).
        rgap = p.gap_mid_r * mm
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            f, "F",
            f"{p.mesh_coarse*mm} - {(p.mesh_coarse-p.mesh_gap)*mm}"
            f"*exp(-((sqrt(x*x+y*y)-{rgap})^2)/(2*{(1.5*mm)**2}))")
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)

        # Extract nodes + all triangles, then tag each triangle by its OWN centroid.
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)[:, :2] / mm     # back to mm for plotting
        idmap = np.full(int(node_tags.max()) + 1, -1, dtype=int)
        idmap[node_tags.astype(int)] = np.arange(len(node_tags))

        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        tris = None
        for et, _, nodes in zip(etypes, etags, enodes):
            if et == 2:                                          # 3-node triangle
                tris = idmap[nodes.reshape(-1, 3).astype(int)]
        if tris is None:
            raise RuntimeError("no triangles in mesh")

        # Weld coincident nodes: gmsh emits duplicates at the 46 arc-centre points
        # and some fragment interfaces, which would float sub-domains and make the
        # stiffness matrix singular. Collapse nodes sharing a position (sub-micron).
        rounded = np.round(node_coords, 5)                       # 1e-5 mm tolerance
        uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        node_coords = uniq
        tris = inv[tris]
        # drop any degenerate triangle (collapsed by the weld)
        good = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
        tris = tris[good]
        # drop unreferenced nodes (the arc-centre points are not in any triangle;
        # left in place they are zero rows -> a singular stiffness matrix)
        used = np.unique(tris)
        remap = np.full(node_coords.shape[0], -1, dtype=int)
        remap[used] = np.arange(len(used))
        node_coords = node_coords[used]
        tris = remap[tris]

        cx = node_coords[tris, 0].mean(axis=1)
        cy = node_coords[tris, 1].mean(axis=1)
        tri_region = np.array([classify(x, y, p, rotor_angle) for x, y in zip(cx, cy)],
                              dtype=int)

        return {
            "p": np.ascontiguousarray(node_coords.T),   # 2 x Npts (mm)
            "t": np.ascontiguousarray(tris.T),           # 3 x Ntri
            "region": tri_region,                        # Ntri
            "rotor_angle": rotor_angle,
        }
    finally:
        gmsh.finalize()


def render_mesh(mesh: dict, p: MotorParams, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    pts, tris, region = mesh["p"], mesh["t"], mesh["region"]
    verts = pts[:, tris.T]                       # 2 x Ntri x 3
    verts = np.transpose(verts, (1, 2, 0))       # Ntri x 3 x 2
    colors = [REGION_COLORS[r] for r in region]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=130)
    pc = PolyCollection(verts, facecolors=colors, edgecolors="#00000022", linewidths=0.12)
    ax.add_collection(pc)
    R = p.rotor_outer_r * 1.04
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect("equal"); ax.axis("off")
    ntri = tris.shape[1]
    ax.set_title(f"{p.ratio_note} outrunner — {ntri} elements\n"
                 f"Ø{2*p.rotor_outer_r:.0f} mm × {p.stack_len:.0f} mm  ·  gap {p.air_gap} mm",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=REGION_COLORS[r]) for r in REGION_NAMES]
    ax.legend(handles, [REGION_NAMES[r] for r in REGION_NAMES],
              loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return ntri


# ---------------------------------------------------------------------------
# Winding — automatic phase/sign per slot via the "star of slots" (EMF phasor)
# method, so any feasible slots/poles combo gets a sensible concentrated winding.
# ---------------------------------------------------------------------------
def winding_table(p: MotorParams):
    """Return (phase[slot], sign[slot]) for a double-layer concentrated winding."""
    Q, pp = p.slots, p.pole_pairs
    phase = np.zeros(Q, dtype=int)
    sign = np.zeros(Q, dtype=int)
    for k in range(Q):
        ang = (k * pp * tau / Q) % tau           # electrical angle of this slot's EMF
        # 6 sectors of 60 deg -> phase A(+/-), C(-/+), B(+/-) ... standard mapping
        sector = int(ang // (pi / 3)) % 6
        phase[k], sign[k] = {
            0: (0, +1), 1: (2, -1), 2: (1, +1),
            3: (0, -1), 4: (2, +1), 5: (1, -1),
        }[sector]
    return phase, sign


# ---------------------------------------------------------------------------
# Magnetostatic solve (vector potential A_z), scikit-fem.
#   weak form:  ∫ ν ∇A·∇v dΩ = ∫ Jz v dΩ + ∫ ν (Brx ∂y v − Bry ∂x v) dΩ
#   PMs enter as the ν·Br source; iron is linear (μr) for now.
# ---------------------------------------------------------------------------
def _element_geometry(pm, t):
    x = pm[0, t]; y = pm[1, t]                   # (3, Ntri), metres
    cx = x.mean(axis=0); cy = y.mean(axis=0)
    area = 0.5 * np.abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    rc = np.hypot(cx, cy); th = np.arctan2(cy, cx)
    return cx, cy, rc, th, area


def material_arrays(mesh, p: MotorParams, phase_currents=(0.0, 0.0, 0.0)):
    pm = mesh["p"] * 1e-3                          # mm -> m
    t = mesh["t"]; region = mesh["region"]
    rot = mesh["rotor_angle"]
    cx, cy, rc, th, area = _element_geometry(pm, t)
    n = t.shape[1]

    mur = np.ones(n)
    mur[(region == STATOR_IRON) | (region == ROTOR_IRON)] = p.iron_mur_linear
    mur[region == MAGNET] = p.magnet_mur
    nu = 1.0 / (MU0 * mur)

    # PM remanence vector (radial, polarity alternating per pole)
    Brx = np.zeros(n); Bry = np.zeros(n)
    mag = region == MAGNET
    pole_pitch = tau / p.poles
    k = np.round((th[mag] - rot) / pole_pitch).astype(int) % p.poles
    pol = np.where(k % 2 == 0, 1.0, -1.0)
    Brx[mag] = p.magnet_Br * pol * np.cos(th[mag])
    Bry[mag] = p.magnet_Br * pol * np.sin(th[mag])

    # Current density per coil element from the winding table
    Jz = np.zeros(n)
    if any(abs(c) > 0 for c in phase_currents):
        phase, sign = winding_table(p)
        slot_pitch = tau / p.slots
        coil = region == COIL
        slot_k = (np.round(th[coil] / slot_pitch).astype(int)) % p.slots
        # slot copper area per slot index
        slot_area = np.zeros(p.slots)
        np.add.at(slot_area, slot_k, area[coil])
        ph = phase[slot_k]; sg = sign[slot_k]
        Ivec = np.array(phase_currents)
        NI = p.turns_per_coil * Ivec[ph] * sg
        Jz[coil] = NI / slot_area[slot_k]
    return nu, Brx, Bry, Jz, (cx, cy, rc, th, area)


def solve_magnetostatic(mesh, p: MotorParams, phase_currents=(0.0, 0.0, 0.0)):
    from skfem import Basis, ElementTriP1, MeshTri, BilinearForm, LinearForm, condense, solve
    from skfem.helpers import dot, grad

    pm = mesh["p"] * 1e-3
    t = mesh["t"]
    m = MeshTri(pm, t)
    basis = Basis(m, ElementTriP1())
    nqp = basis.X.shape[1]

    nu, Brx, Bry, Jz, geom = material_arrays(mesh, p, phase_currents)
    bc = lambda a: np.broadcast_to(a[:, None], (a.shape[0], nqp))

    @BilinearForm
    def a_form(u, v, w):
        return w["nu"] * dot(grad(u), grad(v))

    @LinearForm
    def l_form(v, w):
        gx, gy = grad(v)
        return w["Jz"] * v + w["nu"] * (w["Brx"] * gy - w["Bry"] * gx)

    K = a_form.assemble(basis, nu=bc(nu))
    f = l_form.assemble(basis, nu=bc(nu), Brx=bc(Brx), Bry=bc(Bry), Jz=bc(Jz))

    ro = p.rotor_outer_r * 1e-3
    D = basis.get_dofs(lambda x: np.hypot(x[0], x[1]) > ro - 1e-4)
    Az = solve(*condense(K, f, D=D))

    # Per-element flux density from the nodal potential: B=(∂A/∂y, −∂A/∂x)
    g = basis.interpolate(Az).grad
    gx = g[0].mean(axis=1); gy = g[1].mean(axis=1)
    Bx = gy; By = -gx
    Bmag = np.hypot(Bx, By)

    # Air-gap torque (Arkkio): T = L/(μ0(ro−ri)) ∫ r·Br·Bθ dS over the gap annulus
    cx, cy, rc, th, area = geom
    gapmask = mesh["region"] == AIRGAP
    cth, sth = np.cos(th), np.sin(th)
    Br = Bx * cth + By * sth
    Bt = -Bx * sth + By * cth
    r_i = p.stator_or * 1e-3; r_o = p.magnet_inner_r * 1e-3
    L = p.stack_len * 1e-3
    torque = (L / (MU0 * (r_o - r_i))) * np.sum(
        (rc * Br * Bt * area)[gapmask])

    return {"Az": Az, "Bx": Bx, "By": By, "Bmag": Bmag,
            "torque": torque, "basis": basis, "geom": geom}


def render_field(mesh, sol, p: MotorParams, path: Path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts = mesh["p"]; tris = mesh["t"].T
    triang = mtri.Triangulation(pts[0], pts[1], tris)
    Bmag = np.clip(sol["Bmag"], 0, np.percentile(sol["Bmag"], 99.5))

    fig, ax = plt.subplots(figsize=(7.4, 7.4), dpi=130)
    tpc = ax.tripcolor(triang, facecolors=Bmag, cmap="inferno", shading="flat")
    # flux lines = contours of A_z
    ax.tricontour(triang, sol["Az"], levels=28, colors="#39c0ff",
                  linewidths=0.45, alpha=0.7)
    R = p.rotor_outer_r * 1.02
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("|B|  [T]")
    ax.set_title(title or f"{p.ratio_note} — flux density & lines\n"
                 f"peak |B| {sol['Bmag'].max():.2f} T   torque {sol['torque']*1e3:+.1f} mN·m",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver studies: cogging (unpowered holding) and Kt (powered torque constant).
# ---------------------------------------------------------------------------
def cogging_sweep(p: MotorParams, n: int = 13):
    """Torque vs rotor angle over ONE cogging period, no current = holding torque."""
    period = tau / np.lcm(p.slots, p.poles)              # mechanical radians
    angles = np.linspace(0.0, period, n)
    torque = np.empty(n)
    for i, a in enumerate(angles):
        mesh = build_mesh(p, rotor_angle=a)
        torque[i] = solve_magnetostatic(mesh, p)["torque"]
    return np.degrees(angles), torque, np.degrees(period)


def kt_estimate(p: MotorParams, n: int = 18):
    """Sweep the 3-phase current vector angle at a fixed rotor pos; peak torque
    over the sweep is the q-axis (max) torque -> Kt = T_peak / I_peak."""
    mesh = build_mesh(p, rotor_angle=0.0)
    gammas = np.linspace(0.0, tau, n, endpoint=False)
    I = p.phase_current
    torque = np.empty(n)
    for i, g in enumerate(gammas):
        cur = (I * cos(g), I * cos(g - tau / 3), I * cos(g + tau / 3))
        torque[i] = solve_magnetostatic(mesh, p, phase_currents=cur)["torque"]
    return np.degrees(gammas), torque


def render_curves(cog, kt, p: MotorParams, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cog_a, cog_t, period = cog
    kt_g, kt_t = kt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)

    ax1.plot(cog_a, cog_t * 1e3, "o-", color="#c0392b")
    ax1.axhline(0, color="k", lw=0.5)
    pp = (cog_t.max() - cog_t.min()) * 1e3
    ax1.set_title(f"Cogging (unpowered) — ±{pp/2:.2f} mN·m\nperiod {period:.2f}° mech")
    ax1.set_xlabel("rotor angle [° mech]"); ax1.set_ylabel("torque [mN·m]")
    ax1.grid(alpha=0.3)

    Tpk = np.abs(kt_t).max()
    Kt = Tpk / p.phase_current
    ax2.plot(kt_g, kt_t * 1e3, "o-", color="#1f6f8b")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_title(f"Torque vs current angle @ {p.phase_current:.0f} A\n"
                  f"peak {Tpk*1e3:.0f} mN·m  →  Kt ≈ {Kt:.3f} N·m/A")
    ax2.set_xlabel("current vector angle [° elec]"); ax2.set_ylabel("torque [mN·m]")
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return pp, Tpk, Kt


if __name__ == "__main__":
    p = MotorParams()
    do_curves = len(sys.argv) > 1 and sys.argv[1] == "curves"
    print(f"Building {p.ratio_note} outrunner mesh "
          f"(Ø{2*p.rotor_outer_r:.0f} mm, gap {p.air_gap} mm) ...")
    mesh = build_mesh(p, rotor_angle=0.0)
    ntri = mesh["t"].shape[1]
    npts = mesh["p"].shape[1]
    counts = {REGION_NAMES[r]: int((mesh["region"] == r).sum()) for r in REGION_NAMES}
    print(f"  nodes={npts}  elements={ntri}")
    for name, c in counts.items():
        print(f"    {name:12s} {c}")
    render_mesh(mesh, p, OUT / "mesh.png")
    print(f"  wrote {OUT/'mesh.png'}")

    print("Solving magnetostatic field (PMs only, no current) ...")
    sol = solve_magnetostatic(mesh, p)
    print(f"  peak |B| = {sol['Bmag'].max():.2f} T")
    print(f"  cogging torque at this position = {sol['torque']*1e3:+.2f} mN·m")
    render_field(mesh, sol, p, OUT / "field.png")
    print(f"  wrote {OUT/'field.png'}")

    if do_curves:
        print("Cogging sweep (no current, one cogging period) ...")
        cog = cogging_sweep(p)
        print("Kt sweep (rated current, vary current-vector angle) ...")
        kt = kt_estimate(p)
        pp, Tpk, Kt = render_curves(cog, kt, p, OUT / "curves.png")
        print(f"  wrote {OUT/'curves.png'}")
        eta, ratio = 0.90, 10                       # cycloidal stage (cycloidal-center)
        print("\n  ── custom actuator summary (motor × cycloidal 10:1) ──")
        print(f"   motor Kt              ≈ {Kt:.3f} N·m/A   (= Ke in V·s/rad)")
        print(f"   motor peak torque     ≈ {Tpk*1e3:.0f} mN·m  @ {p.phase_current:.0f} A")
        print(f"   cogging (unpowered)   ≈ ±{pp/2:.2f} mN·m")
        print(f"   → joint output torque ≈ {Tpk*ratio*eta:.2f} N·m  (×{ratio} ×{eta:.0%})")
        print(f"   → joint holding (cog) ≈ {pp/2*ratio*1e-3:.3f} N·m unpowered detent")
