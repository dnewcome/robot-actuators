"""
Kinematics + statics of a SERVO-DRIVEN, TENDON-PULLED SEGMENTED TENDRIL — the buildable 1-DOF
gripper finger. Rigid TPU VERTEBRAE are joined by a thin flexible SPINE; one MG996R hobby servo
(mounted with its shaft ACROSS the finger axis, so the horn spins in the bending plane) reels an
antagonistic string loop straight down the axis — no 90° cable bend.

Why segmented (vs one solid flexing beam):
    bending concentrates at the thin spine sections between rigid vertebrae, so the finger is a
    string of discrete flexure hinges — predictable, and it maps 1:1 onto the sim's hinge chain.
    The rigid vertebrae also house the tendon channels at a generous CONSTANT offset d, which
    dodges the thin-tip-vs-fat-channel conflict of a tapered solid beam.

Why one servo drives a clean 1-DOF bend:
    the two strings sit at ±d from the neutral (spine) axis. Pulling one side by ΔL bends every
    joint toward it; the other side lengthens by the same ΔL — exactly what the horn pays out. The
    horn spins in the bending plane with the strings tangent to it, so take-up is ΔL = r_h·ψ (a
    little capstan) and the antagonistic pair is kinematically self-consistent: one ψ ↔ one bend.

Model — discrete flexure-jointed spine:
    joint i   thin spine flexure, length = gap, section w_spine×t_spine
              rotational stiffness  k = E·I_spine/gap ,  I_spine = w_spine·t_spine³/12
    moment    each joint sees T·d from the tendon → bends Δθ = T·d/k
    the SAME two sums drive everything (as the continuous J1/J2 did, now discrete):
        θ  = T·J1 ,  J1 = Σ d/k = N_j·d/k          (tip angle)
        ΔL = T·J2 ,  J2 = Σ d²/k = N_j·d²/k         (tendon take-up)
    drive     ΔL = r_h·ψ  →  T = r_h·ψ/J2  →  dθ/dψ = r_h/d_eff,  d_eff ≡ J2/J1 = d (constant offset)

Forces:  isometric tip-push F = surplus_T/(d|tip|/dΔL) [virtual work];  wrap/contact F ≈ T·θ [capstan].
Trustworthy: transmission, the discrete curl shape, tension↔servo-torque, the wrap force, the
vertebra-collision curl limit. First-order: linear TPU spine, rigid vertebrae, straight tendon in
the vertebrae. E_tpu is the dominant uncertainty — everything scales ∝ 1/E.

    ../.venv/bin/python tendril/tendril.py         # report + out/tendril.png
"""

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
KGF_CM = 0.0980665                 # 1 kgf·cm in N·m


@dataclass
class TendrilParams:
    # --- segmented TPU finger: rigid vertebrae + thin flexible spine ---
    n_vert: int = 9                # number of rigid vertebrae
    seg_len_mm: float = 7.0        # rigid vertebra length (along the finger axis)
    gap_mm: float = 3.0            # flexure gap between vertebrae (only the spine spans it)
    spine_t_mm: float = 3.0        # spine thickness (bending direction) — the compliant element
    spine_w_mm: float = 6.0        # spine width
    seg_t_mm: float = 8.0          # vertebra thickness (bending dir) — houses the channels
    seg_w_mm: float = 12.0         # vertebra width
    d_off_mm: float = 2.5          # tendon channel offset from the neutral (spine) axis — CONSTANT
    E_tpu_MPa: float = 25.0        # TPU flexural modulus — DOMINANT uncertainty (only the spine flexes)

    # --- single hobby servo, shaft ACROSS the axis, antagonistic string pair ---
    horn_r_mm: float = 8.0         # string hole radius on the horn (in-plane lever → take-up r·sinψ)
    servo_stall_kgcm: float = 10.0 # MG996R ~10 kgf·cm @6V
    servo_travel_deg: float = 90.0 # ± horn range
    string_break_N: float = 200.0  # Dyneema ~0.5 mm
    tpu_pullout_N: float = 30.0    # practical channel tear-out limit (the real tension ceiling)

    # ---- SI helpers -----------------------------------------------------------
    @property
    def n_joints(self): return self.n_vert - 1
    @property
    def length_mm(self):
        return self.n_vert * self.seg_len_mm + self.n_joints * self.gap_mm
    @property
    def L(self): return self.length_mm * 1e-3
    @property
    def d(self): return self.d_off_mm * 1e-3
    @property
    def r_h(self): return self.horn_r_mm * 1e-3
    @property
    def E(self): return self.E_tpu_MPa * 1e6
    @property
    def tau_stall(self): return self.servo_stall_kgcm * KGF_CM
    @property
    def psi_max(self): return np.deg2rad(self.servo_travel_deg)
    @property
    def T_max(self): return self.tau_stall / self.r_h
    @property
    def T_cap(self):
        """Practical tension ceiling: whichever of servo / string / TPU-channel gives out first."""
        return min(self.T_max, self.string_break_N, self.tpu_pullout_N)

    # ---- spine flexure stiffness ----------------------------------------------
    @property
    def I_spine(self):
        return self.spine_w_mm * 1e-3 * (self.spine_t_mm * 1e-3) ** 3 / 12.0
    @property
    def k_joint(self):
        """Rotational stiffness of ONE spine flexure [N·m/rad]."""
        return self.E * self.I_spine / (self.gap_mm * 1e-3)
    @property
    def EI_vert(self):
        return self.E * self.seg_w_mm * 1e-3 * (self.seg_t_mm * 1e-3) ** 3 / 12.0

    # ---- the two (now discrete) compliance sums --------------------------------
    @property
    def J1(self): return self.n_joints * self.d / self.k_joint          # θ  = T·J1
    @property
    def J2(self): return self.n_joints * self.d ** 2 / self.k_joint     # ΔL = T·J2
    @property
    def d_eff(self): return self.J2 / self.J1                           # = d (constant offset)
    @property
    def ratio(self):
        """Small-signal transmission dθ/dψ AT NEUTRAL = r_h/d_eff. The horn is an in-plane lever
        (string tied to a hole), so take-up is r_h·sinψ → the bend SATURATES toward full travel."""
        return self.r_h / self.d_eff
    @property
    def joint_angle_max(self):
        """Per-joint bend at which adjacent vertebrae collide on the concave edge (hard stop)."""
        return (self.gap_mm * 1e-3) / (self.seg_t_mm * 1e-3 / 2.0)

    # ---- forward map: servo angle → tension → curl -----------------------------
    def delta_L(self, psi):
        """Tendon take-up [m] from the in-plane horn lever: string tied at r_h, so ΔL = r_h·sinψ
        (signed). Saturates at r_h as ψ→90° — the geometric price of no 90° cable bend."""
        return self.r_h * np.sin(psi)

    def tension_for(self, psi):
        """Tendon tension to hold horn angle ψ: |ΔL| = T·J2."""
        return abs(self.delta_L(psi)) / self.J2

    def joint_angle(self, psi):
        """Signed per-joint bend Δθ = (ΔL/d)/N_j  (θ_total = T·J1 = ΔL/d)."""
        return (self.delta_L(psi) / self.d) / self.n_joints

    def _walk(self, psi):
        """Vertebra-boundary node polyline (x along, y lateral) + per-vertebra heading + tip angle."""
        dth = self.joint_angle(psi)
        seg = self.seg_len_mm * 1e-3
        gap = self.gap_mm * 1e-3
        pts = [(0.0, 0.0)]
        heads = []
        ang = 0.0
        x = y = 0.0
        for i in range(self.n_vert):
            heads.append(ang)
            x += seg * np.cos(ang); y += seg * np.sin(ang); pts.append((x, y))
            if i < self.n_vert - 1:
                ang += dth
                x += gap * np.cos(ang); y += gap * np.sin(ang); pts.append((x, y))
        return np.array(pts), np.array(heads), self.n_joints * dth

    def bend_deg(self, psi):
        return np.rad2deg(self.n_joints * self.joint_angle(psi))

    def tip(self, psi):
        pts, _, _ = self._walk(psi)
        return pts[-1, 0], pts[-1, 1]

    @property
    def usable_psi(self):
        """Largest horn angle whose tendon pull stays under T_cap. Pull = r_h·sinψ/J2, so this
        inverts through arcsin (capped at the mechanical travel)."""
        s = self.T_cap * self.J2 / self.r_h
        return self.psi_max if s >= 1.0 else min(self.psi_max, float(np.arcsin(s)))

    @property
    def usable_bend_deg(self):
        return self.bend_deg(self.usable_psi)

    # ---- forces ----------------------------------------------------------------
    def wrap_force(self, psi):
        """Capstan-like total normal force on a grasped object ≈ T·|θ|."""
        return self.tension_for(psi) * abs(np.deg2rad(self.bend_deg(psi)))

    def tip_force(self, psi):
        """Isometric fingertip push (virtual work): F = surplus · dΔL/d|tip|. Uses the ACTUAL
        take-up increment ΔL=r_h·sinψ (both dΔL and d|tip| → 0 together near full travel)."""
        eps = np.deg2rad(0.5)
        x0, y0 = self.tip(psi)
        x1, y1 = self.tip(psi + eps)
        dtip = np.hypot(x1 - x0, y1 - y0)
        dL = abs(self.delta_L(psi + eps) - self.delta_L(psi))
        if dtip <= 0:
            return 0.0
        surplus = max(self.T_cap - self.tension_for(psi), 0.0)
        return surplus * dL / dtip

    def servo_torque_frac(self, psi):
        return self.tension_for(psi) * self.r_h / self.tau_stall

    # ---- validation ------------------------------------------------------------
    def checks(self):
        c = []
        th = self.bend_deg(self.psi_max)
        c.append(("full-travel bend is a useful curl (45–270°)",
                  45 <= th <= 270, f"{th:.0f}° at ψ=±{self.servo_travel_deg:.0f}°"))
        c.append(("transmission ratio sane (horn not over-driving)",
                  0.4 <= self.ratio <= 4.0,
                  f"dθ/dψ = {self.ratio:.2f} (r_h {self.horn_r_mm}/d {self.d_off_mm} mm)"))
        c.append(("spine is the compliant element (≪ vertebra stiffness)",
                  self.I_spine * 8 < self.seg_w_mm * 1e-3 * (self.seg_t_mm * 1e-3) ** 3 / 12.0,
                  f"vertebra EI / spine EI = {self.EI_vert/(self.E*self.I_spine):.0f}×"))
        c.append(("channels fit inside the vertebra thickness",
                  self.d_off_mm + 0.8 + 0.4 < self.seg_t_mm / 2,
                  f"d {self.d_off_mm} + ch 0.8 + wall < t/2 {self.seg_t_mm/2:.1f} mm"))
        c.append(("vertebrae don't collide at full travel (≤ hard stop)",
                  abs(self.joint_angle(self.psi_max)) < self.joint_angle_max,
                  f"{np.rad2deg(abs(self.joint_angle(self.psi_max))):.0f}°/joint < "
                  f"{np.rad2deg(self.joint_angle_max):.0f}° stop"))
        c.append(("tension-limited usable curl is a real grip (≥90°)",
                  self.usable_bend_deg >= 90.0,
                  f"{self.usable_bend_deg:.0f}° at T_cap {self.T_cap:.0f} N (full needs "
                  f"{self.tension_for(self.psi_max):.0f} N)"))
        return c

    @property
    def is_valid(self):
        return all(ok for _, ok, _ in self.checks())


def report(p: TendrilParams | None = None):
    p = p or TendrilParams()
    print("\n" + "=" * 70)
    print("  SEGMENTED SERVO-DRIVEN TENDRIL  —  vertebrae + spine, 1 hobby servo")
    print("=" * 70)
    print(f"  finger: {p.n_vert} vertebrae ({p.seg_len_mm:.0f} mm, {p.seg_t_mm:.0f}×{p.seg_w_mm:.0f}) "
          f"+ {p.n_joints} spine flexures ({p.gap_mm:.0f} mm, {p.spine_t_mm:.0f}×{p.spine_w_mm:.0f}), "
          f"L {p.length_mm:.0f} mm, E {p.E_tpu_MPa:.0f} MPa")
    print(f"  spine flexure k = {p.k_joint*1e3:.1f} mN·m/rad/joint   "
          f"(vertebra is {p.EI_vert/(p.E*p.I_spine):.0f}× stiffer → bends only at the spine)")
    print(f"  drive: horn r {p.horn_r_mm:.0f} mm (shaft across axis, no cable bend), "
          f"MG996R {p.servo_stall_kgcm:.0f} kgf·cm, ±{p.servo_travel_deg:.0f}°")
    print(f"  transmission  dθ/dψ|₀ = {p.ratio:.2f} (saturating, ΔL=r_h·sinψ)   "
          f"(tendon offset d = {p.d_off_mm:.1f} mm)")
    up = p.usable_psi
    print(f"  USABLE curl (tension-limited @ T_cap={p.T_cap:.0f} N): {p.usable_bend_deg:.0f}° at horn "
          f"±{np.rad2deg(up):.0f}°   |   full ±{p.servo_travel_deg:.0f}° → {p.bend_deg(p.psi_max):.0f}° "
          f"needs {p.tension_for(p.psi_max):.0f} N")
    print("-" * 70)
    for frac in (0.4, 0.7, 1.0):
        psi = frac * up
        th = p.bend_deg(psi); tx, ty = p.tip(psi)
        print(f"  horn {np.rad2deg(psi):>3.0f}° → bend {th:5.0f}° ({np.rad2deg(p.joint_angle(psi)):.0f}°/joint) "
              f"| tip ({tx*1e3:5.1f}, {ty*1e3:5.1f}) mm | pull {p.tension_for(psi):4.0f} N "
              f"({p.servo_torque_frac(psi)*100:2.0f}%) | tip {p.tip_force(psi):4.1f} N | wrap {p.wrap_force(psi):4.0f} N")
    print("-" * 70)
    print(f"  practical tension ceiling T_cap = {p.T_cap:.0f} N "
          f"(servo {p.T_max:.0f} / string {p.string_break_N:.0f} / TPU {p.tpu_pullout_N:.0f})")
    print("-" * 70)
    for name, ok, detail in p.checks():
        print(f"    [{'ok ' if ok else 'XX '}] {name:<48} {detail}")
    print(f"  -> {'VALID' if p.is_valid else 'INVALID'}")
    print("=" * 70 + "\n")
    return p


def _vert_polys(p, psi):
    """Corner polygons of each vertebra (for the side-view envelope) at horn angle ψ."""
    pts, heads, _ = p._walk(psi)
    seg = p.seg_len_mm * 1e-3
    half = p.seg_t_mm / 2 * 1e-3
    polys = []
    node = 0
    for i in range(p.n_vert):
        x0, y0 = pts[node]
        ang = heads[i]
        ax, ay = np.cos(ang), np.sin(ang)          # along
        nx, ny = -np.sin(ang), np.cos(ang)         # normal (thickness)
        x1, y1 = x0 + seg * ax, y0 + seg * ay
        poly = np.array([[x0 + nx * half, y0 + ny * half],
                         [x1 + nx * half, y1 + ny * half],
                         [x1 - nx * half, y1 - ny * half],
                         [x0 - nx * half, y0 - ny * half]])
        polys.append(poly * 1e3)
        node += 2                                   # skip the gap node
    return polys


def render(p: TendrilParams | None = None):
    p = p or TendrilParams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig = plt.figure(figsize=(13, 9))
    up = p.usable_psi
    up_deg = np.rad2deg(up)

    # (a) segmented curl — vertebrae as blocks, both directions
    ax0 = fig.add_subplot(2, 2, 1)
    curves = [(0.0, "0.6", 1.0), (up_deg / 2, "C0", 1.0), (up_deg, "C1", 1.0),
              (p.servo_travel_deg, "C3", 0.35)]
    for ad, col, al in curves:
        for sgn in ((1,) if ad == 0 else (1, -1)):
            for poly in _vert_polys(p, np.deg2rad(ad) * sgn):
                ax0.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=0.30 * al,
                                      edgecolor=col, lw=0.8))
    ax0.plot([], [], color="C1", lw=3, label=f"usable ±{up_deg:.0f}° → {p.usable_bend_deg:.0f}°")
    ax0.plot([], [], color="C3", lw=3, alpha=0.4,
             label=f"full ±{p.servo_travel_deg:.0f}° ({p.bend_deg(p.psi_max):.0f}°, >T_cap)")
    ax0.plot(0, 0, "ks", ms=6)
    ax0.set_title(f"(a) segmented curl — {p.n_vert} vertebrae bend at {p.n_joints} spine flexures")
    ax0.set_xlabel("x [mm]"); ax0.set_ylabel("y [mm]")
    ax0.set_aspect("equal"); ax0.autoscale_view(); ax0.grid(alpha=0.3); ax0.legend(fontsize=8)

    # (b) transmission
    ax1 = fig.add_subplot(2, 2, 2)
    psis = np.linspace(0, p.psi_max, 60)
    ax1.plot(np.rad2deg(psis), [p.bend_deg(ps) for ps in psis], "C0", lw=1.8, label="tip bend θ")
    ax1.axvspan(0, up_deg, color="C2", alpha=0.10, label="usable (pull ≤ T_cap)")
    ax1.set_xlabel("servo horn angle ψ [deg]"); ax1.set_ylabel("tip bend θ [deg]")
    ax1.set_title(f"(b) transmission — saturating (ΔL=r_h·sinψ), dθ/dψ|₀ = {p.ratio:.2f}")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(np.rad2deg(psis), [p.tension_for(ps) for ps in psis], "C3", lw=1.4)
    ax1b.axhline(p.T_cap, color="C3", ls=":", lw=1.0)
    ax1b.text(2, p.T_cap + 1, f"T_cap {p.T_cap:.0f} N", color="C3", fontsize=8)
    ax1b.set_ylabel("tendon pull [N]", color="C3"); ax1b.tick_params(axis="y", labelcolor="C3")
    ax1.legend(fontsize=8, loc="upper left")

    # (c) fingertip workspace arc
    ax2 = fig.add_subplot(2, 2, 3)
    pu = np.linspace(-up, up, 80)
    tu = np.array([p.tip(ps) for ps in pu]) * 1e3
    ax2.plot(tu[:, 0], tu[:, 1], "C1", lw=2.2, label=f"usable arc (±{up_deg:.0f}°)")
    pf = np.linspace(-p.psi_max, p.psi_max, 100)
    tf = np.array([p.tip(ps) for ps in pf]) * 1e3
    ax2.plot(tf[:, 0], tf[:, 1], "C3", lw=1.0, ls="--", label="full travel (>T_cap)")
    ax2.scatter([0], [p.length_mm], c="0.5", s=25, zorder=5, label="straight tip")
    ax2.plot(0, 0, "ks", ms=6, label="root")
    ax2.set_title("(c) fingertip workspace — 1-DOF arc, ± both ways")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    # (d) grip forces vs bend
    ax3 = fig.add_subplot(2, 2, 4)
    psis3 = np.linspace(np.deg2rad(3), p.psi_max, 60)
    bd = [p.bend_deg(ps) for ps in psis3]
    ax3.plot(bd, [p.tip_force(ps) for ps in psis3], "C0", lw=1.7, label="isometric tip force")
    ax3.plot(bd, [p.wrap_force(ps) for ps in psis3], "C3", lw=1.7, label="wrap / contact force (T·θ)")
    ax3.axvline(p.usable_bend_deg, color="0.4", ls=":", lw=1.0)
    ax3.text(p.usable_bend_deg - 3, ax3.get_ylim()[1] * 0.5, "usable limit",
             rotation=90, va="center", ha="right", fontsize=8, color="0.4")
    ax3.set_xlabel("tip bend θ [deg]"); ax3.set_ylabel("force [N]")
    ax3.set_title("(d) grip forces vs curl — weak tip-push, strong wrap-squeeze")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8, loc="upper left")

    fig.suptitle("Segmented servo-driven tendril — rigid vertebrae + flexible spine, one hobby "
                 "servo (shaft across the axis)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "tendril.png", dpi=120)
    print(f"  wrote {OUT/'tendril.png'}")


if __name__ == "__main__":
    p = report()
    render(p)
