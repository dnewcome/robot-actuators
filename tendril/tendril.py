"""
Kinematics + statics of a SERVO-DRIVEN, TENDON-PULLED CONTINUUM TENDRIL — the buildable
1-DOF cousin of the flex/ gimbal. A tapered flexible TPU beam bolts (M3) into a rigid PLA
mount; ONE hobby servo (MG996R) pulls an ANTAGONISTIC string pair straight off its horn.

Why one servo drives a clean 1-DOF bend:
    the two strings attach at radius r_h on OPPOSITE sides of the horn. Rotate the horn by ψ
    and one string is reeled in by r_h·ψ while the other pays out by r_h·ψ. Inside the beam the
    tendons sit at offset ±d(s) from the neutral axis, so pulling one side by ΔL bends the beam
    (concave toward the pulled tendon). The convex side lengthens by the SAME ΔL — which is
    exactly what the horn pays out. So the antagonistic pair is kinematically self-consistent:
    one horn angle ψ ↔ one bend, both directions (±ψ), no slack, no second motor.

Model — planar tapered elastica (constant tension along a straight-ish channel):
    section     rectangular w(s)×t(s), both taper base→tip; bends in the THICKNESS plane
                EI(s) = E · w(s)·t(s)³/12      (E = TPU flexural modulus — the big unknown)
    tendon      offset d(s) = off_frac·t(s)/2 from the neutral axis
    curvature   κ(s) = T·d(s)/EI(s)            (tension T from the pulled tendon)
    two compliance integrals set everything:
        θ  = T·J1 ,   J1 = ∫ d/EI ds           (tip tangent / total bend)
        ΔL = T·J2 ,   J2 = ∫ d²/EI ds          (tendon take-up)
    drive       ΔL = r_h·ψ  →  T = r_h·ψ/J2  →  dθ/dψ = r_h·J1/J2 = r_h/d_eff  (transmission)
                d_eff ≡ J2/J1 is the LENGTH-WEIGHTED tendon offset (tip-dominated when floppy).

Because the beam TAPERS, EI collapses toward the tip, so curvature concentrates there and the
tendril curls TIGHTER at the tip — the elephant-trunk / SpiRobs curl that makes it grip.

Forces (two honest facets):
    isometric tip force   F_tip = surplus_tension / (d|tip|/dΔL)   [virtual work]
    wrap / contact force  F_wrap ≈ T·θ                            [capstan effect on the channel]
    surplus_tension = T_cap − T_hold(pose);  T_cap = min(servo, string, TPU pull-out).

Trustworthy: the transmission ratio, the curl shape / workspace arc, tension ↔ servo-torque, the
capstan wrap force. First-order: linear (not hyperelastic) TPU modulus, straight tendon channels
(no friction), point contact. E_tpu is the dominant uncertainty — everything scales ∝ 1/E.

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
    # --- flexible TPU beam (tapered rectangular section, bends in the thickness plane) ---
    length_mm: float = 85.0
    w_base_mm: float = 12.0        # width at the bolt flange
    w_tip_mm: float = 6.0          # width at the tip
    t_base_mm: float = 7.5         # thickness (bending direction) at the flange — houses the channels
    t_tip_mm: float = 3.0          # thickness at the tip — thin, soft, curls freely
    E_tpu_MPa: float = 8.0         # TPU ~85A (soft) flexural modulus — DOMINANT uncertainty (5–30 MPa)
    off_frac: float = 0.40         # tendon channel offset = off_frac·(t/2) from neutral axis
    tendon_frac: float = 0.75      # tendon drives the proximal fraction; distal tip is a passive tail

    # --- single hobby servo, antagonistic string pair off the horn ---
    horn_r_mm: float = 2.5         # string attach radius on the servo horn (tendon travel = r·ψ)
    servo_stall_kgcm: float = 10.0 # MG996R ~10 kgf·cm @6V
    servo_travel_deg: float = 90.0 # ± horn range
    string_break_N: float = 200.0  # Dyneema ~0.5 mm
    tpu_pullout_N: float = 30.0    # practical channel tear-out limit (the real tension ceiling)

    n_seg: int = 160

    # ---- SI helpers -----------------------------------------------------------
    @property
    def L(self): return self.length_mm * 1e-3
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
    @property
    def s_term(self):
        """Arc length where the tendon terminates (channels end); tip beyond is a passive tail."""
        return self.tendon_frac * self.L

    # ---- tapered section --------------------------------------------------------
    def _s(self):
        return np.linspace(0.0, self.L, self.n_seg)

    def w(self, s):
        return (self.w_base_mm + (self.w_tip_mm - self.w_base_mm) * (s / self.L)) * 1e-3

    def t(self, s):
        return (self.t_base_mm + (self.t_tip_mm - self.t_base_mm) * (s / self.L)) * 1e-3

    def EI(self, s):
        return self.E * self.w(s) * self.t(s) ** 3 / 12.0

    def d(self, s):
        """Tendon offset from the neutral axis — off_frac·(t/2) over the driven length, 0 past
        the termination (the passive tail carries no tendon, so drives no curvature)."""
        return np.where(np.asarray(s) <= self.s_term, self.off_frac * self.t(s) / 2.0, 0.0)

    # ---- the two compliance integrals ------------------------------------------
    def _integrals(self):
        s = self._s()
        ei = self.EI(s); dd = self.d(s)
        _trap = getattr(np, "trapezoid", None) or np.trapz
        J1 = _trap(dd / ei, s)             # θ  = T·J1
        J2 = _trap(dd ** 2 / ei, s)        # ΔL = T·J2
        return J1, J2

    @property
    def d_eff(self):
        """Length-weighted tendon offset J2/J1 — the effective moment arm of the drive."""
        J1, J2 = self._integrals()
        return J2 / J1

    @property
    def ratio(self):
        """Transmission dθ/dψ = r_h·J1/J2 = r_h/d_eff  [rad bend per rad horn]."""
        return self.r_h / self.d_eff

    # ---- forward map: servo angle → tension → curl shape -----------------------
    def tension_for(self, psi):
        """Tendon tension to hold horn angle ψ (position control): ΔL = r_h·ψ = T·J2."""
        _, J2 = self._integrals()
        return abs(self.r_h * psi) / J2

    def shape(self, psi):
        """Return (s, theta, x, y, kappa, T) for the curled centerline at horn angle ψ."""
        s = self._s()
        sgn = np.sign(psi) if psi != 0 else 1.0
        T = self.tension_for(psi)
        kappa = sgn * T * self.d(s) / self.EI(s)
        ds = np.diff(s)
        theta = np.concatenate([[0.0], np.cumsum((kappa[1:] + kappa[:-1]) / 2 * ds)])
        x = np.concatenate([[0.0], np.cumsum((np.cos(theta[1:]) + np.cos(theta[:-1])) / 2 * ds)])
        y = np.concatenate([[0.0], np.cumsum((np.sin(theta[1:]) + np.sin(theta[:-1])) / 2 * ds)])
        return s, theta, x, y, kappa, T

    def bend_deg(self, psi):
        _, theta, _, _, _, _ = self.shape(psi)
        return np.rad2deg(theta[-1])

    @property
    def usable_psi(self):
        """Largest horn angle whose tendon pull stays under the channel/string ceiling T_cap.
        Tension is linear in ψ, so this is a closed form (capped at the mechanical travel)."""
        _, J2 = self._integrals()
        return min(self.psi_max, self.T_cap * J2 / self.r_h)

    @property
    def usable_bend_deg(self):
        return self.bend_deg(self.usable_psi)

    def tip(self, psi):
        _, _, x, y, _, _ = self.shape(psi)
        return x[-1], y[-1]

    # ---- forces ----------------------------------------------------------------
    def wrap_force(self, psi):
        """Capstan-like total normal force the tendon presses on its channel ≈ T·|θ| — this is
        what squeezes an object the tendril has curled around."""
        _, theta, _, _, _, T = self.shape(psi)
        return T * abs(theta[-1])

    def tip_force(self, psi):
        """Isometric fingertip push force (virtual work): surplus tension / (tip travel per unit
        tendon travel). Rises as the tendril curls (mechanical advantage) until tension runs out."""
        eps = np.deg2rad(0.5)
        x0, y0 = self.tip(psi)
        x1, y1 = self.tip(psi + eps)
        dtip = np.hypot(x1 - x0, y1 - y0)
        dL = self.r_h * eps
        G = dtip / dL if dL > 0 else np.inf          # d|tip| / dΔL
        surplus = max(self.T_cap - self.tension_for(psi), 0.0)
        return surplus / G if G > 0 else 0.0

    def servo_torque_frac(self, psi):
        """Fraction of the servo's stall torque consumed just holding this bend."""
        return self.tension_for(psi) * self.r_h / self.tau_stall

    # ---- validation ------------------------------------------------------------
    def checks(self):
        c = []
        th = self.bend_deg(self.psi_max)
        c.append(("full-travel bend is a useful curl (45–270°)",
                  45 <= th <= 270, f"{th:.0f}° at ψ=±{self.servo_travel_deg:.0f}°"))
        c.append(("transmission ratio sane (horn not over-driving)",
                  0.4 <= self.ratio <= 4.0, f"dθ/dψ = {self.ratio:.2f} (r_h {self.horn_r_mm}/d_eff {self.d_eff*1e3:.1f} mm)"))
        c.append(("taper collapses EI toward the tip (curls at tip)",
                  self.EI(np.array([self.L]))[0] < 0.4 * self.EI(np.array([0.0]))[0],
                  f"EI tip/base = {self.EI(np.array([self.L]))[0]/self.EI(np.array([0.0]))[0]:.2f}"))
        c.append(("tension-limited usable curl is a real grip (≥90°)",
                  self.usable_bend_deg >= 90.0,
                  f"{self.usable_bend_deg:.0f}° at T_cap {self.T_cap:.0f} N (full travel would need "
                  f"{self.tension_for(self.psi_max):.0f} N)"))
        st = self.s_term
        d_st = self.off_frac * self.t(st) / 2.0 * 1e3          # offset at the termination (mm)
        t_half = self.t(st) / 2.0 * 1e3
        c.append(("Ø1.6 channel + wall fits at the tendon termination",
                  d_st + 0.8 + 0.4 < t_half,
                  f"d {d_st:.1f} + ch 0.8 + wall 0.4 < t/2 {t_half:.1f} mm @ {self.tendon_frac:.0%} L"))
        return c

    @property
    def is_valid(self):
        return all(ok for _, ok, _ in self.checks())


def report(p: TendrilParams | None = None):
    p = p or TendrilParams()
    J1, J2 = p._integrals()
    print("\n" + "=" * 70)
    print("  SERVO-DRIVEN CONTINUUM TENDRIL  —  1 hobby servo, antagonistic strings")
    print("=" * 70)
    print(f"  TPU beam  L {p.length_mm:.0f} mm,  w {p.w_base_mm:.0f}→{p.w_tip_mm:.0f},  "
          f"t {p.t_base_mm:.0f}→{p.t_tip_mm:.0f} mm,  E {p.E_tpu_MPa:.0f} MPa")
    print(f"  EI  base {p.EI(np.array([0.0]))[0]*1e3:.1f} → tip {p.EI(np.array([p.L]))[0]*1e3:.2f} mN·m²  "
          f"({p.EI(np.array([0.0]))[0]/p.EI(np.array([p.L]))[0]:.0f}× stiffer at the root)")
    print(f"  drive: horn r {p.horn_r_mm:.0f} mm, MG996R {p.servo_stall_kgcm:.0f} kgf·cm "
          f"(stall τ {p.tau_stall:.2f} N·m), ±{p.servo_travel_deg:.0f}°")
    print(f"  transmission  dθ/dψ = {p.ratio:.2f}   (effective offset d_eff = {p.d_eff*1e3:.1f} mm)")
    up = p.usable_psi
    print(f"  USABLE curl (tension-limited @ T_cap={p.T_cap:.0f} N): {p.usable_bend_deg:.0f}° at horn "
          f"±{np.rad2deg(up):.0f}°   |   full ±{p.servo_travel_deg:.0f}° travel → {p.bend_deg(p.psi_max):.0f}° "
          f"but needs {p.tension_for(p.psi_max):.0f} N")
    print("-" * 70)
    for frac in (0.4, 0.7, 1.0):
        psi = frac * up
        th = p.bend_deg(psi); tx, ty = p.tip(psi)
        print(f"  horn {np.rad2deg(psi):>3.0f}° → bend {th:5.0f}° | tip ({tx*1e3:5.1f}, {ty*1e3:5.1f}) mm | "
              f"pull {p.tension_for(psi):4.0f} N ({p.servo_torque_frac(psi)*100:2.0f}% of stall) | "
              f"tip-force {p.tip_force(psi):4.1f} N | wrap {p.wrap_force(psi):4.0f} N")
    print("-" * 70)
    print(f"  practical tension ceiling T_cap = {p.T_cap:.0f} N "
          f"(servo {p.T_max:.0f} / string {p.string_break_N:.0f} / TPU {p.tpu_pullout_N:.0f})")
    print("-" * 70)
    for name, ok, detail in p.checks():
        print(f"    [{'ok ' if ok else 'XX '}] {name:<44} {detail}")
    print(f"  -> {'VALID' if p.is_valid else 'INVALID'}")
    print("=" * 70 + "\n")
    return p


def render(p: TendrilParams | None = None):
    p = p or TendrilParams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9))
    up = p.usable_psi
    up_deg = np.rad2deg(up)
    s_term_mm = p.s_term * 1e3

    # (a) curl shapes with thickness envelope, both bend directions
    ax0 = fig.add_subplot(2, 2, 1)
    curves = [(0.0, "0.6", "-", "straight"),
              (up_deg / 2, "C0", "-", None),
              (up_deg, "C1", "-", f"usable ±{up_deg:.0f}° → {p.usable_bend_deg:.0f}°"),
              (p.servo_travel_deg, "C3", "--", f"full ±{p.servo_travel_deg:.0f}° "
               f"({p.bend_deg(p.psi_max):.0f}°, >T_cap)")]
    for ad, col, ls, lab in curves:
        for sgn in ((1,) if ad == 0 else (1, -1)):
            s, th, x, y, _, _ = p.shape(np.deg2rad(ad) * sgn)
            nx, ny = -np.sin(th), np.cos(th)
            half = p.t(s) / 2
            ex, ey = x + nx * half, y + ny * half
            fx, fy = x - nx * half, y - ny * half
            if ls == "-":
                ax0.fill(np.concatenate([ex, fx[::-1]]) * 1e3,
                         np.concatenate([ey, fy[::-1]]) * 1e3, color=col, alpha=0.28, lw=0)
            ax0.plot(x * 1e3, y * 1e3, color=col, lw=1.4, ls=ls,
                     label=(lab if sgn == 1 else None))
            # mark where the tendon terminates (passive tail beyond)
            k = int(np.argmin(np.abs(s - p.s_term)))
            ax0.plot(x[k] * 1e3, y[k] * 1e3, "o", color=col, ms=3)
    ax0.plot(0, 0, "ks", ms=6)
    ax0.set_title(f"(a) curl — driven to {p.tendon_frac:.0%}L (dots), passive tail beyond")
    ax0.set_xlabel("x [mm]"); ax0.set_ylabel("y [mm]")
    ax0.set_aspect("equal"); ax0.grid(alpha=0.3); ax0.legend(fontsize=8)

    # (b) transmission: bend angle & required pull vs horn angle
    ax1 = fig.add_subplot(2, 2, 2)
    psis = np.linspace(0, p.psi_max, 60)
    ax1.plot(np.rad2deg(psis), [p.bend_deg(ps) for ps in psis], "C0", lw=1.8, label="tip bend θ")
    ax1.axvspan(0, up_deg, color="C2", alpha=0.10, label="usable (pull ≤ T_cap)")
    ax1.set_xlabel("servo horn angle ψ [deg]"); ax1.set_ylabel("tip bend θ [deg]")
    ax1.set_title(f"(b) transmission — linear, dθ/dψ = {p.ratio:.2f}")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(np.rad2deg(psis), [p.tension_for(ps) for ps in psis], "C3", lw=1.4, label="tendon pull")
    ax1b.axhline(p.T_cap, color="C3", ls=":", lw=1.0)
    ax1b.text(2, p.T_cap + 1, f"T_cap {p.T_cap:.0f} N", color="C3", fontsize=8)
    ax1b.set_ylabel("tendon pull [N]", color="C3"); ax1b.tick_params(axis="y", labelcolor="C3")
    ax1.legend(fontsize=8, loc="upper left")

    # (c) tip workspace arc (both directions) — usable solid, full-travel dashed
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

    # (d) grip forces vs bend — the torque-budget trade
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

    fig.suptitle("Servo-driven continuum tendril — 1 hobby servo pulls an antagonistic "
                 "string pair on a tapered TPU beam", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "tendril.png", dpi=120)
    print(f"  wrote {OUT/'tendril.png'}")


if __name__ == "__main__":
    p = report()
    render(p)
