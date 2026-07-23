"""
Y-ZIPPER model — now following the PAPER'S actual primitive algorithm.

Ref: Jiaji Li et al., "Y-zipper: 3D Printing Flexible–Rigid Transition Mechanism for
Rapid and Reversible Assembly", ACM CHI '26. doi:10.1145/3772318.3790723.
(Appendix A.3 gives the primitive formulas; §3.1 the tooth/ball-socket geometry.)

An earlier version of this file used a made-up "differential pitch → radius" rule and
uniform rectangular teeth. That is NOT how the paper programs shapes. The real method:

TOOTH GEOMETRY (§3.1): each vertical segment is three teeth a,b,c with a WAVE-LIKE
profile that cyclically overlap (left of a over right of b, b over c, c over a) — a
mutually-supporting ring. Each tooth carries a BALL NODE (∅1=2.4 mm) on top and a
SOCKET (∅2=3.0 mm) below (tol 0.3 mm) for shear/alignment. TPU/PLA compliant BRIDGES
(thickness D) join segments and carry the tension when zipped.

PROGRAMMING SHAPES = varying the teeth, not the pitch (Appendix A.3):
  • Straight — uniform teeth; slider runs straight up z.            z = v·t
  • Bend    — teeth at interface α–β are THINNER than at α–γ. Accumulate the two
              interface tooth-thickness sums T1, T2 over the prism:
                  θ = 2(T1 − T2)/(√3·w),     R = (T1 + T2)/(2θ)                 (eqs 1-2)
              (differential accumulated THICKNESS across the width w — a bimetallic-strip
               bend, the interface separation being the triangle height √3·w/2.)
  • Screw   — shift the LOWER side of each tooth up by Δt; the ball-node normal rotates:
                  θ_z = k·Δt   (linear, −1.5t ≤ Δt ≤ 2.5t),
                  L_z = √(L² − (r·θ_z)²),   r = w/√3                            (eq 7)
  • Coil    — bend PLUS an axial rise Δz per segment → a true (non-planar) helix
              of radius R and pitch h.                                          (eqs 4-6)
  • Series  — primitives compose by accumulating homogeneous transforms.        (eq 11)

STIFFNESS is taken from the paper's MEASURED three-point-bending values (not an idealized
closed-section calc, which overpredicts because teeth+bridges ≠ a solid tube):
  EI_strip ≈ 1.9e3 N·mm² (unzipped)  →  EI_rod ≈ 3.1e5 N·mm² (zipped)  ≈ 160× stiffer.

This module builds each primitive's swept 3D geometry from those equations (reproducing
the paper's Fig. 6) and reports the calibrated stiffness / load numbers.

    ../.venv/bin/python yzipper/yzipper.py        # report + out/primitives.png
"""

from dataclasses import dataclass, field
from math import pi, sqrt, sin, cos, radians, degrees
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


# --------------------------------------------------------------------------- #
# homogeneous-transform helpers (frame accumulation, paper eq. 11)
# --------------------------------------------------------------------------- #
def _trans(x, y, z):
    M = np.eye(4); M[:3, 3] = (x, y, z); return M

def _rotx(a):
    c, s = cos(a), sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def _rotz(a):
    c, s = cos(a), sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _integrate(increments, frame0=None):
    """Walk a list of (dθx, dθz, ds) increments into a list of 4×4 frames.
    dθx = local bending (about x, curvature), dθz = local twist (about z, torsion),
    ds = advance along the local axis. This is the paper's per-segment composition."""
    frame = np.eye(4) if frame0 is None else frame0.copy()
    frames = [frame.copy()]
    for dtx, dtz, ds in increments:
        frame = frame @ _rotx(dtx) @ _rotz(dtz) @ _trans(0, 0, ds)
        frames.append(frame.copy())
    return frames


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass
class YZipperParams:
    # --- strip / section (paper symbols) ---------------------------------------
    width_w_mm: float = 25.0         # strip width w (tent uses 25 mm); triangle side
    tooth_t_mm: float = 1.2          # avg tooth thickness t (min 0.5; <1.2 → ball/socket precision poor)
    tooth_H_mm: float = 6.0          # tooth height H (acts like a shell wall thickness)
    bridge_D_mm: float = 1.2         # compliant-bridge thickness D (TPU 0.6–2.4, PLA 0.4–1.2)
    n_teeth: int = 20                # teeth per segment interface (the 2n in eq 1 → n per interface)

    # --- ball node & socket (§3.1, shear/alignment) ----------------------------
    ball_d1_mm: float = 2.4          # ball-node diameter ∅1
    socket_d2_mm: float = 3.0        # socket diameter ∅2
    tol_mm: float = 0.3              # fit tolerance (TPU 68D → 0.1)

    # --- material ---------------------------------------------------------------
    E_pla_MPa: float = 2600.0        # PLA modulus (paper's value)

    # --- MEASURED stiffness anchors (paper §9.2, three-point bending) -----------
    EI_strip_Nmm2: float = 1.9e3     # unzipped strip bending rigidity
    EI_rod_Nmm2: float = 3.1e5       # zipped rod bending rigidity  (ratio ≈ 160×)
    load_D08_kg: float = 11.0        # max load at bridge D = 0.8 mm
    load_D20_kg: float = 18.0        # max load at bridge D = 2.0 mm

    # --- actuation --------------------------------------------------------------
    closure_speed_mms: float = 250.0     # ~20–30 cm/s
    k_screw_rad_per_mm: float = 0.35     # screw constant k in θ_z = k·Δt  (CALIBRATION-PENDING)

    # ---- derived section geometry --------------------------------------------
    @property
    def r_circum_mm(self) -> float:
        """Triangle circumradius r = w/√3 (paper's r in the screw formula, eq 7)."""
        return self.width_w_mm / sqrt(3.0)

    @property
    def interface_sep_mm(self) -> float:
        """Perpendicular separation of the two bend interfaces = triangle height √3·w/2.
        This is the lever that turns a tooth-thickness difference into a bend angle."""
        return sqrt(3.0) / 2.0 * self.width_w_mm

    # ---- measured stiffness ----------------------------------------------------
    @property
    def stiffness_ratio(self) -> float:
        return self.EI_rod_Nmm2 / self.EI_strip_Nmm2

    def beam_stiffness_N_per_mm(self, span_mm: float, zipped: bool = True) -> float:
        """Three-point-bending stiffness k = 48·EI/L³ (paper §9.2)."""
        EI = self.EI_rod_Nmm2 if zipped else self.EI_strip_Nmm2
        return 48.0 * EI / span_mm**3

    def max_load_kg(self, bridge_D_mm: float) -> float:
        """Linear interpolation of max supported load vs bridge thickness D (§9.2)."""
        m = (self.load_D20_kg - self.load_D08_kg) / (2.0 - 0.8)
        return self.load_D08_kg + m * (bridge_D_mm - 0.8)

    # ---- BEND primitive (eqs 1-2) ---------------------------------------------
    def bend_angle_rad(self, T1_mm: float, T2_mm: float) -> float:
        """θ = 2(T1 − T2)/(√3·w) from the accumulated interface tooth thicknesses."""
        return 2.0 * (T1_mm - T2_mm) / (sqrt(3.0) * self.width_w_mm)

    def bend_radius_mm(self, T1_mm: float, T2_mm: float) -> float:
        """R = (T1 + T2)/(2θ)."""
        th = self.bend_angle_rad(T1_mm, T2_mm)
        return (T1_mm + T2_mm) / (2.0 * th) if th else float("inf")

    def thickness_diff_for_angle(self, theta_rad: float) -> float:
        """DESIGN inverse (what the Grasshopper tool exposes): the accumulated
        thickness difference (T1 − T2) needed for a target bend angle θ."""
        return theta_rad * sqrt(3.0) * self.width_w_mm / 2.0

    # ---- SCREW primitive (eq 7) -----------------------------------------------
    def screw_angle_rad(self, dt_mm: float) -> float:
        """θ_z = k·Δt (linear; valid −1.5t ≤ Δt ≤ 2.5t)."""
        return self.k_screw_rad_per_mm * dt_mm

    def screw_height_mm(self, L_flat_mm: float, dt_mm: float) -> float:
        """L_z = √(L² − (r·θ_z)²), r = w/√3 — vertical height after the twist."""
        thz = self.screw_angle_rad(dt_mm)
        val = L_flat_mm**2 - (self.r_circum_mm * thz) ** 2
        return sqrt(val) if val > 0 else 0.0

    def screw_dt_range_mm(self):
        return (-1.5 * self.tooth_t_mm, 2.5 * self.tooth_t_mm)

    # ---- feasibility -----------------------------------------------------------
    def checks(self):
        out = []
        out.append(("tooth thickness ≥ 0.5 mm (printable) and ≥ 1.2 for ball/socket",
                    self.tooth_t_mm >= 1.2, f"t = {self.tooth_t_mm} mm"))
        out.append(("strip width ≥ 8 mm (teeth interlock)",
                    self.width_w_mm >= 8.0, f"w = {self.width_w_mm} mm"))
        out.append(("ball/socket clearance ∅2−∅1 in [0.2, 0.8] (stable close)",
                    0.2 <= (self.socket_d2_mm - self.ball_d1_mm) <= 0.8,
                    f"∅2−∅1 = {self.socket_d2_mm - self.ball_d1_mm:.1f} mm"))
        out.append(("bridge D in TPU/PLA compliant range (0.4–2.4)",
                    0.4 <= self.bridge_D_mm <= 2.4, f"D = {self.bridge_D_mm} mm"))
        out.append(("stiffness ratio matches paper (~160×)",
                    120 <= self.stiffness_ratio <= 200, f"{self.stiffness_ratio:.0f}×"))
        return out

    @property
    def is_valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks())


# --------------------------------------------------------------------------- #
# primitives → increment lists (Appendix A.3)
# --------------------------------------------------------------------------- #
def inc_straight(L, n=24):
    return [(0.0, 0.0, L / n)] * n

def inc_bend(theta, R, n=48):
    """Planar arc: bend θ about x, arc length R·θ (= (T1+T2)/2)."""
    return [(theta / n, 0.0, R * theta / n)] * n

def inc_screw(theta_z, Lz, n=48):
    """Straight axis of height Lz with the section twisting θ_z about z."""
    return [(0.0, theta_z / n, Lz / n)] * n

def inc_coil(R, pitch_h, turns, n=140):
    """True helix from curvature/torsion of a helix (radius R, pitch h)."""
    c = pitch_h / (2 * pi)
    denom = R**2 + c**2
    kappa = R / denom                        # curvature  → bend about x
    tau = c / denom                          # torsion    → twist about z
    total_len = turns * sqrt((2 * pi * R) ** 2 + pitch_h**2)
    ds = total_len / n
    return [(kappa * ds, tau * ds, ds)] * n


def rails_and_center(frames, w):
    """Sweep the triangular section (circumradius w/√3) along the frames → 3 edge
    rails (α, β, γ) plus the centreline, for plotting a rod like the paper's Fig. 6."""
    Rc = w / sqrt(3.0)
    local = [np.array([Rc * cos(radians(A)), Rc * sin(radians(A)), 0.0, 1.0])
             for A in (90, 210, 330)]
    rails = [[], [], []]
    center = []
    for F in frames:
        center.append(F[:3, 3])
        for k, v in enumerate(local):
            rails[k].append((F @ v)[:3])
    return [np.array(r) for r in rails], np.array(center)


# --------------------------------------------------------------------------- #
# report + Fig-6 reproduction
# --------------------------------------------------------------------------- #
def report(p: YZipperParams) -> None:
    print("=" * 70)
    print("Y-ZIPPER — paper primitive algorithm (CHI '26, Appendix A.3)")
    print("=" * 70)
    print(f"  section .......... equilateral Δ, width w={p.width_w_mm:.0f} mm, "
          f"tooth t={p.tooth_t_mm} mm, H={p.tooth_H_mm} mm, bridge D={p.bridge_D_mm} mm")
    print(f"  ball / socket .... ∅1={p.ball_d1_mm} / ∅2={p.socket_d2_mm} mm (tol {p.tol_mm})")
    print(f"  r = w/√3 ......... {p.r_circum_mm:.2f} mm    interface sep √3·w/2 = {p.interface_sep_mm:.2f} mm")
    print("-" * 70)
    print("  STIFFNESS (paper measured, three-point bending):")
    print(f"    EI_strip = {p.EI_strip_Nmm2:.3g} N·mm²  →  EI_rod = {p.EI_rod_Nmm2:.3g} N·mm²"
          f"   [{p.stiffness_ratio:.0f}× stiffer]")
    print(f"    beam k=48EI/L³ @L=100mm: strip {p.beam_stiffness_N_per_mm(100,False):.2f} "
          f"→ rod {p.beam_stiffness_N_per_mm(100,True):.1f} N/mm")
    print(f"    max load: D=0.8→{p.max_load_kg(0.8):.0f} kg, D=2.0→{p.max_load_kg(2.0):.0f} kg")
    print("-" * 70)
    print("  BEND primitive  θ = 2(T1−T2)/(√3·w),  R = (T1+T2)/(2θ):")
    for (T1, T2) in [(45, 15), (70, 20), (60, 60)]:
        th = p.bend_angle_rad(T1, T2)
        R = p.bend_radius_mm(T1, T2)
        print(f"    T1={T1:3.0f}, T2={T2:3.0f} mm → θ={degrees(th):5.1f}°, "
              f"R={'∞' if R==float('inf') else f'{R:5.1f} mm'}")
    th90 = radians(90)
    print(f"    DESIGN inverse: bend 90° needs (T1−T2) = {p.thickness_diff_for_angle(th90):.1f} mm")
    print("-" * 70)
    lo, hi = p.screw_dt_range_mm()
    print(f"  SCREW primitive  θ_z = k·Δt  (k={p.k_screw_rad_per_mm} rad/mm, "
          f"Δt∈[{lo:.1f},{hi:.1f}] mm):")
    for dt in (p.tooth_t_mm, 2 * p.tooth_t_mm):
        thz = p.screw_angle_rad(dt)
        Lz = p.screw_height_mm(80.0, dt)
        print(f"    Δt={dt:.1f} mm → θ_z={degrees(thz):5.1f}° over L=80 → L_z={Lz:.1f} mm")
    print("-" * 70)
    print("  feasibility:")
    for name, ok, detail in p.checks():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    print(f"  => {'VALID' if p.is_valid else 'INVALID'}")
    print("=" * 70)


def plot_primitives(p: YZipperParams) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    w = p.width_w_mm
    # build the four primitives from the paper equations
    fr_straight = _integrate(inc_straight(70))
    th_b = p.bend_angle_rad(70, 20); R_b = p.bend_radius_mm(70, 20)
    fr_bend = _integrate(inc_bend(th_b, R_b))
    fr_coil = _integrate(inc_coil(R=14, pitch_h=26, turns=2.4))
    thz = 2 * pi          # illustrative one-turn screw (k in θz=k·Δt is calibration-pending)
    Lz = 70.0
    fr_screw = _integrate(inc_screw(thz, Lz))
    # a series composite (straight + 90° bends), like the cube corner
    R90 = 16.0; th90 = radians(90)
    comp = (inc_straight(22, 8) + inc_bend(th90, R90, 24)
            + inc_straight(22, 8) + inc_bend(th90, R90, 24) + inc_straight(22, 8))
    fr_comp = _integrate(comp)

    panels = [("(a) Straight", fr_straight), ("(b) Bend  θ={:.0f}°".format(degrees(th_b)), fr_bend),
              ("(c) Coil  R=14,h=26", fr_coil), ("(d) Screw  1 turn (k pending)", fr_screw),
              ("(e) Series: straight+90° bends", fr_comp)]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig = plt.figure(figsize=(15, 6.2))
    for i, (title, frames) in enumerate(panels):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        rails, center = rails_and_center(frames, w)
        for k, r in enumerate(rails):
            ax.plot(r[:, 0], r[:, 1], r[:, 2], color=colors[k], lw=1.6)
        ax.plot(center[:, 0], center[:, 1], center[:, 2], color="0.4", lw=0.8, ls="--")
        # draw a few triangle cross-sections
        for j in range(0, len(frames), max(1, len(frames) // 6)):
            tri = np.array([rails[k][j] for k in range(3)] + [rails[0][j]])
            ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color="0.6", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_box_aspect((1, 1, 1.6))
        _equal_3d(ax, center, rails)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.suptitle("Y-zipper motion primitives built from the paper's equations "
                 "(Appendix A.3) — cf. Fig. 6", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(exist_ok=True)
    path = OUT / "primitives.png"
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return path


def _equal_3d(ax, center, rails):
    pts = np.vstack([center] + rails)
    mins, maxs = pts.min(0), pts.max(0)
    ctr = (mins + maxs) / 2
    rng = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)


if __name__ == "__main__":
    p = YZipperParams()
    report(p)
    out = plot_primitives(p)
    print(f"wrote {out}")
