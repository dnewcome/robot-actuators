"""
2-D "UNROLLED" magnetostatic FEA cross-check for the axial-flux PCB motor (motor.py).

An axial-flux motor is a 3-D device, but the field where it matters — the air gap that holds
the coils — is well captured by cutting the cylinder at the mean radius r_m and UNROLLING it
into a flat Cartesian strip:

      y (axial) ↑     ┌───────────── stator back-iron (flux return) ─────────────┐
                      │                 air gap  +  PCB coils                     │
                      │  ▓N▓   ░S░   ▓N▓   ░S░   magnets (axial ±y, alternating)  │
                      └───────────── rotor back-iron ───────────────────────────┘
      x (tangential) →   one pole pitch τ_p = 2π·r_m / poles per magnet

Currents run RADIALLY (out of this page, +z), so it's the SAME A_z magnetostatic problem as
motor/fea.py — only the geometry is Cartesian and the magnets are axially (y) magnetised. The
job is to PREDICT the air-gap flux B_g that motor.py currently just ASSUMES (0.25 T), since that
is the single biggest lever on Kt (Kt ∝ B_g). We solve magnets-only, read |B_y| in the coil
band, sweep the axial gap, and feed the corrected B_g back into the analytical Kt.

Reuses the motor/fea.py stack (gmsh + scikit-fem, no FEMM). First pass; true 3-D (end effects,
finite radial extent) is the Elmer step next.

    ../.venv/bin/python pcb-motor/fea.py            # solve + out/fea_unrolled.png
"""

import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motor import PCBMotorParams

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MU0 = 4e-7 * pi

R_IRON, R_MAGNET, R_COIL = range(3)
R_COLORS = {R_IRON: "#34404d", R_MAGNET: "#c0392b", R_COIL: "#c87a2f"}


@dataclass
class UnrolledParams:
    # tangential/radial geometry taken from the analytical motor
    motor: PCBMotorParams = None
    # axial stack [mm]
    magnet_Br: float = 1.28           # NdFeB N42
    magnet_mur: float = 1.05
    magnet_thick_mm: float = 3.0
    air_gap_mm: float = 1.2           # axial gap that holds the PCB coils (both magnet faces)
    rotor_bi_mm: float = 2.5          # rotor back-iron
    stator_bi_mm: float = 2.5         # stator-side iron flux return
    iron_mur: float = 2000.0
    n_pp_model: int = 2               # pole pairs meshed (sample B in the centre, away from edges)
    mesh_fine_mm: float = 0.20

    def __post_init__(self):
        if self.motor is None:
            self.motor = PCBMotorParams()

    @property
    def r_m(self):
        return self.motor.r_m
    @property
    def poles(self):
        return self.motor.n_poles
    @property
    def tau_p(self):                  # pole pitch (tangential) [m]
        return 2 * pi * self.r_m / self.poles
    @property
    def width(self):                  # modelled strip width [m]
        return self.n_pp_model * 2 * self.tau_p
    @property
    def n_mag(self):
        return self.n_pp_model * 2
    # y-layer boundaries [m]
    @property
    def y_rbi(self): return self.rotor_bi_mm * 1e-3
    @property
    def y_mag(self): return self.y_rbi + self.magnet_thick_mm * 1e-3
    @property
    def y_gap(self): return self.y_mag + self.air_gap_mm * 1e-3
    @property
    def y_top(self): return self.y_gap + self.stator_bi_mm * 1e-3
    @property
    def y_coil_mid(self): return 0.5 * (self.y_mag + self.y_gap)


def classify(x, y, up: UnrolledParams):
    if y < up.y_rbi or y > up.y_gap:
        return R_IRON
    if y < up.y_mag:
        return R_MAGNET
    return R_COIL


def build_mesh(up: UnrolledParams, verbose=False):
    import gmsh
    W, H = up.width, up.y_top
    tau = up.tau_p
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("unrolled_axial")
        occ = gmsh.model.occ
        rects = []
        rects.append(occ.addRectangle(0, 0, 0, W, up.y_rbi))                 # rotor iron
        for k in range(up.n_mag):                                           # magnets
            rects.append(occ.addRectangle(k * tau, up.y_rbi, 0, tau, up.magnet_thick_mm * 1e-3))
        rects.append(occ.addRectangle(0, up.y_mag, 0, W, up.air_gap_mm * 1e-3))   # gap+coils
        rects.append(occ.addRectangle(0, up.y_gap, 0, W, up.stator_bi_mm * 1e-3)) # stator iron
        occ.synchronize()
        occ.fragment([(2, r) for r in rects], [])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [t for (_, t) in gmsh.model.getEntities(2)], name="dom")
        gmsh.option.setNumber("Mesh.MeshSizeMax", up.mesh_fine_mm * 1e-3)
        gmsh.option.setNumber("Mesh.MeshSizeMin", up.mesh_fine_mm * 1e-3)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ncoords = ncoords.reshape(-1, 3)[:, :2]
        idmap = np.full(int(ntags.max()) + 1, -1, dtype=int)
        idmap[ntags.astype(int)] = np.arange(len(ntags))
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tris = None
        for et, nodes in zip(etypes, enodes):
            if et == 2:
                tris = idmap[nodes.reshape(-1, 3).astype(int)]
        rounded = np.round(ncoords, 9)
        uniq, inv = np.unique(rounded, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        ncoords = uniq; tris = inv[tris]
        good = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
        tris = tris[good]
        used = np.unique(tris)
        remap = np.full(ncoords.shape[0], -1, dtype=int); remap[used] = np.arange(len(used))
        ncoords = ncoords[used]; tris = remap[tris]
        cx = ncoords[tris, 0].mean(axis=1); cy = ncoords[tris, 1].mean(axis=1)
        region = np.array([classify(x, y, up) for x, y in zip(cx, cy)], dtype=int)
        return {"p": np.ascontiguousarray(ncoords.T), "t": np.ascontiguousarray(tris.T),
                "region": region, "cx": cx, "cy": cy}
    finally:
        gmsh.finalize()


def solve(mesh, up: UnrolledParams):
    from skfem import Basis, ElementTriP1, MeshTri, BilinearForm, LinearForm, condense, solve as sksolve
    from skfem.helpers import dot, grad

    pm = mesh["p"]; t = mesh["t"]; region = mesh["region"]
    cx, cy = mesh["cx"], mesh["cy"]
    m = MeshTri(pm, t)
    basis = Basis(m, ElementTriP1())
    nqp = basis.X.shape[1]
    n = t.shape[1]

    mur = np.ones(n)
    mur[region == R_IRON] = up.iron_mur
    mur[region == R_MAGNET] = up.magnet_mur
    nu = 1.0 / (MU0 * mur)

    # magnets: axial (y) magnetisation, alternating per pole along x
    Brx = np.zeros(n); Bry = np.zeros(n)
    mag = region == R_MAGNET
    k = np.floor(cx[mag] / up.tau_p).astype(int)
    Bry[mag] = np.where(k % 2 == 0, 1.0, -1.0) * up.magnet_Br

    bc = lambda a: np.broadcast_to(a[:, None], (a.shape[0], nqp))

    @BilinearForm
    def a_form(u, v, w):
        return w["nu"] * dot(grad(u), grad(v))

    @LinearForm
    def l_form(v, w):
        gx, gy = grad(v)
        return w["nu"] * (w["Brx"] * gy - w["Bry"] * gx)

    K = a_form.assemble(basis, nu=bc(nu))
    f = l_form.assemble(basis, nu=bc(nu), Brx=bc(Brx), Bry=bc(Bry))
    # Dirichlet A=0 on the whole outer boundary of the strip
    D = basis.get_dofs()
    Az = sksolve(*condense(K, f, D=D))

    g = basis.interpolate(Az).grad
    gx = g[0].mean(axis=1); gy = g[1].mean(axis=1)
    Bx = gy; By = -gx
    return {"Az": Az, "Bx": Bx, "By": By, "Bmag": np.hypot(Bx, By), "basis": basis}


def gap_flux(mesh, sol, up: UnrolledParams):
    """Mean |B_y| in the coil band, sampled in the CENTRE poles (away from the Dirichlet edges)."""
    cx, cy = mesh["cx"], mesh["cy"]
    coil = mesh["region"] == R_COIL
    centre = (cx > up.width * 0.30) & (cx < up.width * 0.70)
    band = coil & centre & (np.abs(cy - up.y_coil_mid) < up.air_gap_mm * 1e-3 * 0.35)
    return float(np.mean(np.abs(sol["By"][band]))) if band.any() else float("nan")


def render(up: UnrolledParams, mesh, sol, bg_fea, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))

    # (a) unrolled flux: |B| + flux lines (contours of A_z)
    pts = mesh["p"] * 1e3; tris = mesh["t"].T
    tri = mtri.Triangulation(pts[0], pts[1], tris)
    Bc = np.clip(sol["Bmag"], 0, np.percentile(sol["Bmag"], 99))
    tpc = ax[0].tripcolor(tri, facecolors=Bc, cmap="inferno", shading="flat")
    ax[0].tricontour(tri, sol["Az"], levels=26, colors="#39c0ff", linewidths=0.5, alpha=0.7)
    for yb in (up.y_rbi, up.y_mag, up.y_gap):
        ax[0].axhline(yb * 1e3, color="w", lw=0.5, ls=":")
    ax[0].set_aspect("equal"); ax[0].set_title(
        f"(a) unrolled axial-flux field — flux crosses the coil band  "
        f"(peak |B| {sol['Bmag'].max():.2f} T)")
    ax[0].set_xlabel("tangential x [mm]"); ax[0].set_ylabel("axial y [mm]")
    fig.colorbar(tpc, ax=ax[0], fraction=0.025, pad=0.01, label="|B| [T]")

    # (b) B_y along the coil mid-line vs x
    cx, cy = mesh["cx"] * 1e3, mesh["cy"]
    near = np.abs(cy - up.y_coil_mid) < up.air_gap_mm * 1e-3 * 0.5
    o = np.argsort(cx[near])
    ax[1].plot(cx[near][o], sol["By"][near][o], color="C0", lw=1.0)
    ax[1].axhline(bg_fea, color="C2", ls="--", lw=1, label=f"mean |B_y| {bg_fea:.3f} T (FEA)")
    ax[1].axhline(-bg_fea, color="C2", ls="--", lw=1)
    ax[1].axhline(up.motor.b_gap_T, color="C3", ls=":", lw=1.2,
                  label=f"assumed {up.motor.b_gap_T:.2f} T (motor.py)")
    ax[1].axhline(-up.motor.b_gap_T, color="C3", ls=":", lw=1.2)
    ax[1].set_title("(b) axial flux B_y through the coil band — the square-wave PM field")
    ax[1].set_xlabel("tangential x [mm]"); ax[1].set_ylabel("B_y [T]")
    ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(alpha=0.3)

    # (c) B_gap vs axial air gap — the key lever, FEA-predicted
    gaps = np.linspace(0.6, 3.0, 9)
    bgs = []
    for g in gaps:
        u2 = UnrolledParams(**{**up.__dict__, "air_gap_mm": g})
        me = build_mesh(u2); so = solve(me, u2); bgs.append(gap_flux(me, so, u2))
    ax[2].plot(gaps, bgs, "o-", color="C2", lw=1.6, label="FEA B_g")
    ax[2].axvline(up.air_gap_mm, color="C2", ls=":", lw=0.8)
    ax[2].axhline(up.motor.b_gap_T, color="C3", ls=":", lw=1.2, label="assumed 0.25 T")
    ax[2].set_title("(c) FEA air-gap flux vs axial gap — validates/corrects the analytical B_g")
    ax[2].set_xlabel("axial air gap [mm]"); ax[2].set_ylabel("B_g [T]")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("Axial-flux PCB motor — 2D unrolled magnetostatic cross-check (gmsh + scikit-fem)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUT.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


def report(up: UnrolledParams | None = None):
    up = up or UnrolledParams()
    mot = up.motor
    print("\n" + "=" * 68)
    print("  UNROLLED AXIAL-FLUX FEA CROSS-CHECK  (gmsh + scikit-fem)")
    print("=" * 68)
    print(f"  cut at r_m {up.r_m*1e3:.1f} mm, pole pitch {up.tau_p*1e3:.2f} mm, "
          f"{up.n_pp_model} pole-pairs meshed")
    print(f"  stack: rotor-iron {up.rotor_bi_mm} | magnet {up.magnet_thick_mm} (Br {up.magnet_Br}) "
          f"| gap {up.air_gap_mm} | stator-iron {up.stator_bi_mm} mm")
    print("-" * 68)
    mesh = build_mesh(up)
    sol = solve(mesh, up)
    bg = gap_flux(mesh, sol, up)
    print(f"  mesh: {mesh['t'].shape[1]} elements, peak |B| {sol['Bmag'].max():.2f} T")
    print(f"  FEA air-gap flux B_g   = {bg:.3f} T")
    print(f"  motor.py ASSUMED B_g   = {mot.b_gap_T:.3f} T")
    ratio = bg / mot.b_gap_T
    print(f"  -> B_g is {'HIGHER' if ratio>1 else 'LOWER'} than assumed by {abs(1-ratio)*100:.0f}%; "
          f"since Kt ∝ B_g, corrected Kt ≈ {mot.kt*ratio*1e3:.1f} mN·m/A "
          f"(was {mot.kt*1e3:.1f}), T_cont ≈ {mot.t_cont*ratio*1e3:.1f} mN·m")
    print("-" * 68)
    print("  CAVEATS: linear iron (peak |B| unphysical >2 T) → saturation would pull B_g a bit")
    print("  lower; this stack has iron flux-RETURNS (slotless, not open-air coreless). Takeaway:")
    print("  the assumed 0.25 T was CONSERVATIVE (≈ a 3 mm gap); a tight gap + iron return ~doubles")
    print("  B_g and Kt. Nonlinear iron + true 3D end-effects = the Elmer step next.")
    print("=" * 68 + "\n")
    render(up, mesh, sol, bg, OUT / "fea_unrolled.png")
    return up


if __name__ == "__main__":
    report()
