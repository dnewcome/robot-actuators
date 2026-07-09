"""
Kinematics + statics of a 2-DOF TENDON-DRIVEN GIMBAL: a pan/tilt platform on a rigid 2-axis
joint, pulled by 3 Dyneema tendons at 120° wound on 3 capstan motors, recentred by a spring.

The mechanism, and why 3 tendons for 2 DOF:
    tendons can only PULL, so you need n+1 = 3 of them to command an n = 2-DOF joint. That
    leaves ONE redundant DOF — a co-contraction / internal-tension mode that produces zero net
    moment. It's not wasted: it sets the joint STIFFNESS independently of the pose (pull all
    three harder → stiffer joint, same tilt). The centering spring supplies a baseline restoring
    moment so the platform returns to upright when the tendons relax.

Model (rigid gimbal, pivot at height h; platform attach + base attach both at 120°):
    pose        u = (ux, uy)   LEAN VECTOR — tilt of magnitude β=|u| toward physical azimuth
                               atan2(uy,ux); the normal rotates about the ⟂ horizontal axis (Rodrigues)
    attach pts  P_i = C + R(u)·(r_p cosφ_i, r_p sinφ_i, 0),  C=(0,0,h);  B_i = (r_b cosφ_i, …, 0)
    tendon      û_i = (B_i−P_i)/|B_i−P_i|   (pull direction);  length L_i = |B_i−P_i|
    structure   A_[:,i] = [ (P_i−C) × û_i ]_{x,y}   (2×3 moment-per-unit-tension matrix)
    statics     A·t = M_req(u) ,  t_i ≥ t_min  (taut, pull-only);  M_req = spring (k·β, ⟂ to lean)
                + load  →  2 eqns, 3 unknowns: min-norm particular + λ·null(A) picked to keep taut.
    capstan     tendon travel = r_cap·θ_motor ;  max tension = τ_motor / r_cap

Trustworthy: the tendon geometry, the pull-only tension distribution, the reachable cone, the
co-contraction stiffness, the motor-torque→tilt map. First-order: rigid isotropic centering
spring (k_θ·θ), a point pivot, tendons as straight line segments (no routing friction/capstan
losses beyond the tension cap).

    ../.venv/bin/python flex/flex.py            # report + out/flex.png
"""

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass
class FlexParams:
    n_tendons: int = 3
    platform_r_mm: float = 12.0       # tendon attach radius on the moving platform
    base_r_mm: float = 12.0           # tendon anchor / capstan radius on the base
    pivot_h_mm: float = 30.0          # gimbal pivot height (≈ neutral tendon length)
    max_tilt_deg: float = 35.0        # mechanical gimbal limit

    # --- centering spring (isotropic torsional restoring at the pivot) ---------
    k_spring_Nm_rad: float = 0.085    # restoring moment = k·tilt (sized so the reachable cone
                                      #   is tension-limited, not just the mechanical stop)

    # --- capstan motors --------------------------------------------------------
    capstan_r_mm: float = 5.0         # drum radius: tendon travel = r·θ_motor
    motor_stall_Nm: float = 0.020     # per-motor stall torque → max tendon tension
    min_tension_N: float = 0.20       # keep every tendon taut (pull-only floor)

    # ---- geometry (SI) --------------------------------------------------------
    @property
    def r_p(self): return self.platform_r_mm * 1e-3
    @property
    def r_b(self): return self.base_r_mm * 1e-3
    @property
    def h(self): return self.pivot_h_mm * 1e-3
    @property
    def r_cap(self): return self.capstan_r_mm * 1e-3
    @property
    def max_tension(self): return self.motor_stall_Nm / self.r_cap
    @property
    def min_tension(self): return self.min_tension_N
    @property
    def phis(self):
        return np.arange(self.n_tendons) * 2 * pi / self.n_tendons

    @property
    def C(self):
        return np.array([0.0, 0.0, self.h])

    def _R(self, ux, uy):
        """Rotation for the LEAN VECTOR (ux,uy): rotate the platform normal toward physical
        azimuth atan2(uy,ux) by |u| (rad), about the perpendicular horizontal axis. Rodrigues."""
        beta = np.hypot(ux, uy)
        if beta < 1e-12:
            return np.eye(3)
        ax = np.array([-uy, ux, 0.0]) / beta            # ⟂ to the lean direction
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(beta) * K + (1 - np.cos(beta)) * (K @ K)

    def platform_pts(self, ux, uy):
        """World attach points P_i for lean vector (ux,uy)."""
        R = self._R(ux, uy)
        loc = np.stack([self.r_p * np.cos(self.phis), self.r_p * np.sin(self.phis),
                        np.zeros(self.n_tendons)], axis=1)     # (n,3)
        return self.C + loc @ R.T

    @property
    def base_pts(self):
        return np.stack([self.r_b * np.cos(self.phis), self.r_b * np.sin(self.phis),
                         np.zeros(self.n_tendons)], axis=1)

    def tendon_lengths(self, ux, uy):
        return np.linalg.norm(self.base_pts - self.platform_pts(ux, uy), axis=1)

    @property
    def L0(self):
        return self.tendon_lengths(0.0, 0.0)

    def structure_matrix(self, ux, uy):
        """2×3 A: column i = moment (about base x,y) per unit tension in tendon i."""
        P = self.platform_pts(ux, uy)
        u = self.base_pts - P
        u = u / np.linalg.norm(u, axis=1, keepdims=True)      # pull directions
        arms = P - self.C                                      # moment arms about pivot
        M = np.cross(arms, u)                                  # (n,3) moment per unit tension
        return M[:, :2].T                                      # (2, n)

    def _M_req(self, ux, uy, load_mag):
        """Moment the tendons must supply: overcome the spring (magnitude k·β, ⟂ to lean)
        plus a constant external load in the same rotational sense."""
        beta = np.hypot(ux, uy)
        moment_dir = np.array([-uy, ux])                       # ⟂ to lean, the moment axis
        m = self.k_spring_Nm_rad * moment_dir                  # spring: |m| = k·β
        if beta > 1e-9 and load_mag:
            m = m + load_mag * moment_dir / beta
        return m

    # ---- inverse statics: tensions to HOLD a pose -----------------------------
    def solve_tensions(self, ux, uy, load_mag=0.0):
        """Tendon tensions to hold lean (ux,uy) against the spring (+ external load).
        Returns tensions [N] (all ≥ min_tension, ≤ max_tension) or None if infeasible."""
        A = self.structure_matrix(ux, uy)
        M_req = self._M_req(ux, uy, load_mag)
        t_part = np.linalg.pinv(A) @ M_req                    # min-norm particular solution
        null = np.cross(A[0], A[1])                            # 1-D null space of the 2×3 A
        if np.linalg.norm(null) < 1e-12:
            return None
        null = null / np.linalg.norm(null)
        # pick λ so the lowest tendon sits exactly at min_tension (least co-contraction)
        lo, hi = -np.inf, np.inf
        for tp, nn in zip(t_part, null):
            if abs(nn) < 1e-12:
                if tp < self.min_tension - 1e-9:
                    return None
                continue
            bound = (self.min_tension - tp) / nn
            if nn > 0:
                lo = max(lo, bound)
            else:
                hi = min(hi, bound)
        if lo > hi + 1e-9:
            return None
        lam = lo if np.isfinite(lo) else hi
        t = t_part + lam * null
        if np.any(t > self.max_tension + 1e-9):
            return None
        return t

    def co_contraction_tensions(self, ux, uy, extra_N):
        """Add `extra_N` of co-contraction (internal tension, zero net moment) on top of the
        minimal hold — the redundant DOF that stiffens the joint at the same pose."""
        t = self.solve_tensions(ux, uy)
        if t is None:
            return None
        A = self.structure_matrix(ux, uy)
        null = np.cross(A[0], A[1]); null = null / np.linalg.norm(null)
        null = null * np.sign(null[np.argmin(t)])             # +null = all-positive side
        step = extra_N / np.min(np.abs(null[np.abs(null) > 1e-9]))
        cand = t + step * null
        return cand if np.all(cand <= self.max_tension + 1e-9) else None

    # ---- workspace ------------------------------------------------------------
    def reachable_tilt(self, alpha, load_mag=0.0):
        """Max tilt magnitude β [rad] reachable toward azimuth α, capped by tension or mechanics."""
        prev = 0.0
        for beta in np.linspace(0.005, np.deg2rad(self.max_tilt_deg), 80):
            if self.solve_tensions(beta * np.cos(alpha), beta * np.sin(alpha), load_mag) is None:
                return prev
            prev = beta
        return np.deg2rad(self.max_tilt_deg)

    # ---- stiffness ------------------------------------------------------------
    def joint_stiffness(self, ux, uy, tensions):
        """Rotational stiffness [N·m/rad] = −d(net restoring moment)/d(tilt) along the current
        lean direction (spring + geometric tension term), by numerical differentiation."""
        beta = np.hypot(ux, uy)
        dirn = np.array([ux, uy]) / beta if beta > 1e-9 else np.array([1.0, 0.0])
        mdir = np.array([-dirn[1], dirn[0]])                  # moment axis for this lean
        d = 1e-4

        def net(s):
            b = beta + s
            A = self.structure_matrix(b * dirn[0], b * dirn[1])
            return (A @ tensions) @ mdir - self.k_spring_Nm_rad * b
        return -(net(d) - net(-d)) / (2 * d)

    # ---- validation -----------------------------------------------------------
    def checks(self):
        c = []
        c.append(("3 tendons for 2 DOF (pull-only needs n+1)",
                  self.n_tendons >= 3, f"{self.n_tendons} tendons"))
        held = self.solve_tensions(np.deg2rad(self.max_tilt_deg * 0.8), 0.0)
        c.append(("max-ish tilt is holdable within motor torque",
                  held is not None, "feasible" if held is not None else "INFEASIBLE"))
        c.append(("max tendon tension within motor",
                  self.max_tension > 4 * self.min_tension,
                  f"{self.max_tension:.1f} N (τ {self.motor_stall_Nm*1e3:.0f} mN·m / r {self.capstan_r_mm} mm)"))
        c.append(("neutral tendons taut & sensible length",
                  np.all(self.L0 > 0.5 * self.h),
                  f"L0 = {self.L0[0]*1e3:.0f} mm"))
        c.append(("workspace covers a useful cone",
                  np.rad2deg(self.reachable_tilt(0.0)) > 10,
                  f"{np.rad2deg(self.reachable_tilt(0.0)):.0f}° reachable at α=0"))
        return c

    @property
    def is_valid(self):
        return all(ok for _, ok, _ in self.checks())


def report(p: FlexParams | None = None):
    p = p or FlexParams()
    print("\n" + "=" * 68)
    print("  2-DOF TENDON-DRIVEN GIMBAL  —  3 capstans, sprung center")
    print("=" * 68)
    print(f"  platform Ø{2*p.platform_r_mm:.0f} / base Ø{2*p.base_r_mm:.0f} mm, pivot {p.pivot_h_mm:.0f} mm high")
    print(f"  centering spring {p.k_spring_Nm_rad*1e3:.0f} mN·m/rad, max tilt {p.max_tilt_deg:.0f}°")
    print(f"  capstan Ø{2*p.capstan_r_mm:.0f} mm, motor {p.motor_stall_Nm*1e3:.0f} mN·m "
          f"→ max tension {p.max_tension:.1f} N")
    print(f"  neutral tendon length L0 = {p.L0[0]*1e3:.1f} mm")
    print("-" * 68)
    for beta_deg in (12, 22):
        t = p.solve_tensions(np.deg2rad(beta_deg), 0.0)
        if t is not None:
            print(f"  hold {beta_deg}° tilt toward tendon-1: tensions = "
                  f"[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] N  (spring load {p.k_spring_Nm_rad*np.deg2rad(beta_deg)*1e3:.1f} mN·m)")
            k = p.joint_stiffness(np.deg2rad(beta_deg), 0.0, t)
            print(f"       joint stiffness ≈ {k*1e3:.0f} mN·m/rad")
    print("-" * 68)
    alphas = np.linspace(0, 2 * pi, 24, endpoint=False)
    ws = np.rad2deg([p.reachable_tilt(a) for a in alphas])
    print(f"  workspace: reachable tilt {ws.min():.0f}–{ws.max():.0f}° (3-fold ripple), "
          f"mean {ws.mean():.0f}°")
    print("-" * 68)
    for name, ok, detail in p.checks():
        print(f"    [{'ok ' if ok else 'XX '}] {name:<44} {detail}")
    print(f"  -> {'VALID' if p.is_valid else 'INVALID'}")
    print("=" * 68 + "\n")
    return p


def render(p: FlexParams | None = None):
    p = p or FlexParams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9))

    # (a) tendon lengths vs azimuth at fixed tilt — which tendon shortens
    ax0 = fig.add_subplot(2, 2, 1)
    beta = np.deg2rad(18)
    alphas = np.linspace(0, 2 * pi, 200)
    Ls = np.array([p.tendon_lengths(beta * np.cos(a), beta * np.sin(a)) for a in alphas]) * 1e3
    for i in range(p.n_tendons):
        ax0.plot(np.rad2deg(alphas), Ls[:, i], lw=1.5, label=f"tendon {i+1} (@{i*120}°)")
    ax0.axhline(p.L0[0] * 1e3, color="k", ls=":", lw=0.8, label="neutral L0")
    ax0.set_title("(a) tendon lengths vs tilt azimuth (β=18°) — pull-to-bend")
    ax0.set_xlabel("tilt direction α [deg]"); ax0.set_ylabel("tendon length [mm]")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

    # (b) workspace: reachable tilt cone (polar), no load vs a tip load
    ax1 = fig.add_subplot(2, 2, 2, projection="polar")
    al = np.linspace(0, 2 * pi, 96)
    w0 = np.rad2deg([p.reachable_tilt(a, 0.0) for a in al])
    wl = np.rad2deg([p.reachable_tilt(a, 0.010) for a in al])   # 10 mN·m tip load
    ax1.plot(al, w0, color="C0", lw=1.8, label="no load")
    ax1.plot(al, wl, color="C3", lw=1.5, label="+10 mN·m load")
    ax1.set_title("(b) reachable tilt cone [deg] — 3-fold ripple", pad=18)
    ax1.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.15, -0.05))

    # (c) tension distribution around a commanded circle at β=25°
    ax2 = fig.add_subplot(2, 2, 3)
    alc = np.linspace(0, 2 * pi, 200)
    T = []
    for a in alc:
        t = p.solve_tensions(beta * np.cos(a), beta * np.sin(a))
        T.append(t if t is not None else [np.nan] * 3)
    T = np.array(T)
    for i in range(p.n_tendons):
        ax2.plot(np.rad2deg(alc), T[:, i], lw=1.5, label=f"tendon {i+1}")
    ax2.axhline(p.min_tension_N, color="k", ls=":", lw=0.8, label="min (taut)")
    ax2.set_title("(c) tendon tensions around a β=18° circle — pull-only, always taut")
    ax2.set_xlabel("tilt direction α [deg]"); ax2.set_ylabel("tension [N]")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # (d) co-contraction stiffens the joint at fixed pose (the redundant DOF)
    ax3 = fig.add_subplot(2, 2, 4)
    extras = np.linspace(0, p.max_tension - p.min_tension - 0.5, 20)
    ks = []
    for e in extras:
        t = p.co_contraction_tensions(np.deg2rad(15), 0.0, e)
        ks.append(p.joint_stiffness(np.deg2rad(15), 0.0, t) * 1e3 if t is not None else np.nan)
    ax3.plot(extras, ks, "o-", color="C4", lw=1.6)
    ax3.axhline(p.k_spring_Nm_rad * 1e3, color="k", ls=":", lw=0.8, label="spring alone")
    ax3.set_title("(d) co-contraction stiffens the joint at fixed pose (redundant DOF)")
    ax3.set_xlabel("added co-contraction tension [N]"); ax3.set_ylabel("joint stiffness [mN·m/rad]")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle("2-DOF tendon-driven gimbal — 3 capstans pull a sprung pan/tilt platform",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "flex.png", dpi=120)
    print(f"  wrote {OUT/'flex.png'}")


if __name__ == "__main__":
    p = report()
    render(p)
