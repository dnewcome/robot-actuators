"""
Control proof for a CoreXY stage driven by cheap TT gearmotors and closed on PER-AXIS
DRAW-WIRE encoders (drawwire.py) instead of the motor shafts.

Why CoreXY makes the "precision lives in the sensor" argument inescapable:
the two fixed motors are COUPLED into Cartesian motion —
      X = (a + b)/2,   Y = (a − b)/2      (a,b = the two belt feeds)
so every belt defect (backlash, position-dependent stretch, a skipped tooth) leaks into
BOTH X and Y at the toolhead, and the long CoreXY belt path has more of all of them than a
single axis. Motor encoders are structurally blind to it. Close the loop on a load-side
sensor (a string encoder on each axis) and it all lands inside the loop and cancels.

The math also DECOUPLES the dynamics, which is why this reuses the rail-servo drive twice.
In belt coordinates p = X+Y, q = X−Y the toolhead kinetic energy splits,
      KE = ½m(Ẋ²+Ẏ²) = ½(m/2)ṗ² + ½(m/2)q̇²,
so p and q are two INDEPENDENT 1-D drives (each: motor → backlash gap ±δ → belt spring →
load mass m/2), exactly like linear-rail-servo/servo.py. The controller re-couples them:
it reads the toolhead (X,Y), runs a Cartesian PID + profile feedforward, and maps the
command back to the two motors through the CoreXY Jacobian transpose (u_a = Fx+Fy,
u_b = Fx−Fy). Feedforward is done in belt coordinates so each drive is unburdened.

  FEEDBACK MODES:
    string — read (X,Y) from the two draw-wire encoders (σ ≈ drawwire.noise_rms, ~3 µm).
             Backlash + stretch are INSIDE the loop → cancelled at the toolhead.
    shaft  — read (X,Y) from the motor encoders: X=(a_m+b_m)/2 etc. Blind to the belts →
             the toolhead traces a backlash-notched, stretch-bowed path.

Trustworthy: the qualitative belt-error cancellation, the reversal notches ≈ backlash, the
load-dependent bow from position-dependent belt stretch. First-order: lumped equal X/Y
inertia (clean p,q decoupling), smoothed friction, one belt spring per axis, no motor
electrical dynamics.

    ../.venv/bin/python corexy/corexy.py            # report + out/corexy.png
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drawwire import DrawWireEncoder

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


@dataclass
class CoreXYParams:
    # --- moving mass / workspace ----------------------------------------------
    toolhead_mass_kg: float = 0.40     # gantry + toolhead, lumped equal in X and Y
    work_mm: float = 200.0             # square workspace side
    pulley_radius_mm: float = 6.0      # GT2 16T-ish

    # --- TT gearmotor per belt (same garbage as the rail) ----------------------
    out_stall_torque_Nm: float = 0.06
    out_noload_rpm: float = 200.0
    out_inertia: float = 4.0e-5
    f_coulomb: float = 0.10            # gearbox Coulomb, belt-referred [N]
    backlash_deg: float = 3.0          # lash at each output shaft
    cog_frac: float = 0.03
    cog_teeth: float = 12.0

    # --- belt (position-dependent stretch is the CoreXY-specific effect) -------
    belt_k0: float = 9.0e4             # stiffness at the home corner [N/m]
    belt_soften: float = 0.7           # k(x) = k0/(1 + soften·reach/work): far corner softer
    belt_damping: float = 30.0
    c_load: float = 8.0                # viscous drag on the moving mass [N·s/m]
    f_stick: float = 0.3               # Coulomb on the moving mass [N]

    # --- controller (Cartesian PID + belt-coord feedforward) -------------------
    kp: float = 8000.0
    ki: float = 5000.0
    kd: float = 150.0
    d_filt_hz: float = 30.0
    in_pos_um: float = 10.0
    control_hz: float = 1000.0
    shaft_enc_res_mm: float = 0.05

    # --- sim -------------------------------------------------------------------
    dt: float = 2.0e-4
    wire: DrawWireEncoder = field(default_factory=lambda: DrawWireEncoder(travel_mm=300.0))

    # ---- derived --------------------------------------------------------------
    @property
    def r(self):
        return self.pulley_radius_mm * 1e-3

    @property
    def work(self):
        return self.work_mm * 1e-3

    @property
    def motor_mass(self):
        return self.out_inertia / self.r ** 2

    @property
    def f_stall(self):
        return self.out_stall_torque_Nm / self.r

    @property
    def v_noload(self):
        return (self.out_noload_rpm * 2 * np.pi / 60.0) * self.r

    @property
    def backlash(self):
        return np.deg2rad(self.backlash_deg) * self.r

    @property
    def string_sigma(self):
        return self.wire.noise_rms * 1e-3   # mm -> m


def _deadband(e, d):
    return max(0.0, e - d) + min(0.0, e + d)


def simulate(pr: CoreXYParams, path, mode="string", t_end=8.0, seed=0, load=(0.0, 0.0)):
    """Integrate the two decoupled belt drives under Cartesian feedback. `path(t)->(X,Y)` [m].
    `load` is a constant external Cartesian force [N] (to expose belt stretch)."""
    rng = np.random.default_rng(seed)
    dt = pr.dt
    n = int(t_end / dt)
    ctrl_every = max(1, int(round(1.0 / (pr.control_hz * dt))))
    M_m = pr.motor_mass
    M_l = pr.toolhead_mass_kg / 2.0            # effective load mass per belt coord
    b_emf = pr.f_stall / pr.v_noload
    db, cb = pr.backlash, pr.belt_damping
    veps = 1e-4
    a_d = np.exp(-2 * np.pi * pr.d_filt_hz / pr.control_hz)
    i_max = 2 * pr.f_stall / pr.ki
    e_dead = (pr.in_pos_um * 1e-6, pr.in_pos_um * 1e-6) if mode == "string" \
        else (0.7 * pr.shaft_enc_res_mm * 1e-3,) * 2

    # belt-coordinate state: index 0 = p (=X+Y), 1 = q (=X−Y)
    X0, Y0 = path(0.0)
    xx = [X0 + Y0, X0 - Y0]                     # load-side belt coords
    xd = [0.0, 0.0]
    m = list(xx)                               # motor-side belt coords
    md = [0.0, 0.0]
    integ = [0.0, 0.0]; e_prev = [0.0, 0.0]; deriv = [0.0, 0.0]; u = [0.0, 0.0]

    out = {k: np.empty(n) for k in ("t", "X", "Y", "Xc", "Yc", "Xm", "Ym")}

    def belt_k(load_pos):
        reach = abs(load_pos)                  # softer as the belt pays out from home
        return pr.belt_k0 / (1.0 + pr.belt_soften * reach / pr.work)

    for i in range(n):
        t = i * dt
        Xc, Yc = path(t)
        # actual toolhead from the load-side belt coords
        X = 0.5 * (xx[0] + xx[1]); Y = 0.5 * (xx[0] - xx[1])

        if i % ctrl_every == 0:
            if mode == "string":               # per-axis draw-wire on the load
                Xmeas = X + rng.normal(0, pr.string_sigma)
                Ymeas = Y + rng.normal(0, pr.string_sigma)
            else:                              # motor encoders: believe X=(a_m+b_m)/2
                q = pr.shaft_enc_res_mm * 1e-3
                am = np.round(m[0] / q) * q; bm = np.round(m[1] / q) * q
                Xmeas = 0.5 * (am + bm); Ymeas = 0.5 * (am - bm)
            # Cartesian PID per axis
            u_fb = [0.0, 0.0]
            F = [0.0, 0.0]
            for ax, (err_now, meas, cmd) in enumerate(
                    ((Xc - Xmeas, Xmeas, Xc), (Yc - Ymeas, Ymeas, Yc))):
                e = err_now
                deriv[ax] = a_d * deriv[ax] + (1 - a_d) * (e - e_prev[ax]) * pr.control_hz
                e_prev[ax] = e
                if abs(e) < e_dead[ax]:
                    integ[ax] *= 0.98
                    F[ax] = pr.kd * deriv[ax] + pr.ki * integ[ax]
                else:
                    pd = pr.kp * e + pr.kd * deriv[ax]
                    if abs(pd + pr.ki * integ[ax]) < 2 * pr.f_stall:
                        integ[ax] = np.clip(integ[ax] + e / pr.control_hz, -i_max, i_max)
                    F[ax] = pd + pr.ki * integ[ax]
            Fx, Fy = F
            # belt-coordinate feedforward (unburden each drive)
            h = 1.0 / pr.control_hz
            Xp, Yp = path(t + h); Xm2, Ym2 = path(t - h)
            pr_ref = [(Xc + Yc), (Xc - Yc)]
            vref = [((Xp + Yp) - (Xm2 + Ym2)) / (2 * h), ((Xp - Yp) - (Xm2 - Ym2)) / (2 * h)]
            aref = [((Xp + Yp) - 2 * pr_ref[0] + (Xm2 + Ym2)) / (h * h),
                    ((Xp - Yp) - 2 * pr_ref[1] + (Xm2 - Ym2)) / (h * h)]
            # Jacobian transpose: motor a<-Fx+Fy (drives p), motor b<-Fx-Fy (drives q)
            u_fb = [Fx + Fy, Fx - Fy]
            for j in range(2):
                u[j] = u_fb[j] + (M_m + M_l) * aref[j] + b_emf * vref[j]

        # external Cartesian load -> belt coords
        lb = [load[0] + load[1], load[0] - load[1]]
        for j in range(2):
            cog = pr.cog_frac * pr.f_stall * np.sin(pr.cog_teeth * m[j] / pr.r)
            f_motor = np.clip(u[j], -pr.f_stall, pr.f_stall) - b_emf * md[j] + cog \
                - pr.f_coulomb * np.tanh(md[j] / veps)
            e_gap = m[j] - xx[j]
            k = belt_k(xx[j] - (X0 + Y0 if j == 0 else X0 - Y0))
            F_c = k * _deadband(e_gap, db) + (cb * (md[j] - xd[j]) if abs(e_gap) > db else 0.0)
            a_m = (f_motor - F_c) / M_m
            a_x = (F_c - pr.c_load * xd[j] - pr.f_stick * np.tanh(xd[j] / veps) - lb[j]) / M_l
            md[j] += a_m * dt; m[j] += md[j] * dt
            xd[j] += a_x * dt; xx[j] += xd[j] * dt

        out["t"][i] = t; out["Xc"][i] = Xc; out["Yc"][i] = Yc
        out["X"][i] = X; out["Y"][i] = Y
        out["Xm"][i] = 0.5 * (m[0] + m[1]); out["Ym"][i] = 0.5 * (m[0] - m[1])
    return out


# ---- paths ------------------------------------------------------------------
def circle_path(cx, cy, R, period, t0=0.4):
    def f(t):
        if t < t0:
            return cx + R, cy
        ph = 2 * np.pi * (t - t0) / period
        return cx + R * np.cos(ph), cy + R * np.sin(ph)
    return f


def square_path(cx, cy, half, period, t0=0.4):
    pts = [(cx + half, cy + half), (cx - half, cy + half),
           (cx - half, cy - half), (cx + half, cy - half)]

    def f(t):
        if t < t0:
            return pts[0]
        ph = ((t - t0) / period) % 1.0
        seg = int(ph * 4) % 4
        fr = ph * 4 - int(ph * 4)
        a = pts[seg]; b = pts[(seg + 1) % 4]
        return a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr
    return f


CIRCLE_PERIOD = 12.0                       # slow enough for the weak motor (no saturation)


def _trace_error(pr, path, mode, t_end, **kw):
    r = simulate(pr, path, mode, t_end, **kw)
    keep = r["t"] > CIRCLE_PERIOD * 0.6
    ex = r["X"][keep] - r["Xc"][keep]; ey = r["Y"][keep] - r["Yc"][keep]
    return np.hypot(ex, ey)


def hold_droop(pr, mode, Xs, cy, load):
    """Hold at each X under a constant Cartesian `load`; return the settled position error
    [m]. Reveals position-dependent belt stretch: the shaft loop droops (and the droop
    varies with position as the belt softens); the string loop holds regardless."""
    errs = []
    for X in Xs:
        r = simulate(pr, lambda t: (X, cy), mode, 4.5, load=load)   # settle fully under load
        k = r["t"] > 4.0
        errs.append(np.hypot(np.mean(r["X"][k]) - X, np.mean(r["Y"][k]) - cy))
    return np.array(errs)


def report(pr: CoreXYParams | None = None):
    pr = pr or CoreXYParams()
    print("\n" + "=" * 70)
    print("  CoreXY STAGE  —  TT gearmotors, loop on draw-wire scales vs motor shafts")
    print("=" * 70)
    print(f"  toolhead {pr.toolhead_mass_kg*1e3:.0f} g, workspace {pr.work_mm:.0f}² mm, "
          f"pulley Ø{2*pr.pulley_radius_mm:.0f} mm")
    print(f"  per-belt backlash {pr.backlash*1e6:.0f} µm, motor f_stall {pr.f_stall:.1f} N")
    print(f"  draw-wire scale noise σ = {pr.string_sigma*1e6:.1f} µm  "
          f"(shaft encoder LSB {pr.shaft_enc_res_mm*1e3:.0f} µm)")
    print("-" * 70)
    cx = cy = pr.work_mm / 2 * 1e-3
    cir = circle_path(cx, cy, 0.06, period=CIRCLE_PERIOD)
    Xs = np.linspace(cx - 0.06, cx + 0.06, 7)
    for mode in ("shaft", "string"):
        e = _trace_error(pr, cir, mode, t_end=2.5 * CIRCLE_PERIOD) * 1e6
        droop = hold_droop(pr, mode, Xs, cy, load=(0.0, 5.0)) * 1e6
        print(f"  loop on {mode:>6}:  circle RMS {np.sqrt(np.mean(e**2)):6.1f} µm "
              f"(peak {np.max(e):6.1f})   | 5 N hold droop {np.mean(droop):6.1f} µm")
    print("-" * 70)
    print("  Backlash notches the traced circle and belt stretch droops it under load —")
    print("  both only when the loop closes on the motor shaft, not the toolhead.")
    print("=" * 70 + "\n")
    return pr


def render(pr: CoreXYParams | None = None):
    pr = pr or CoreXYParams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cx = cy = pr.work_mm / 2 * 1e-3
    R = 0.06
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    # (a) traced circle: commanded vs motor-loop vs string-loop
    cir = circle_path(cx, cy, R, period=CIRCLE_PERIOD)
    tend = 2.5 * CIRCLE_PERIOD
    rs = simulate(pr, cir, "shaft", tend)
    rr = simulate(pr, cir, "string", tend)
    keep = rs["t"] > CIRCLE_PERIOD * 0.6
    ax[0, 0].plot((rs["Xc"][keep]) * 1e3, (rs["Yc"][keep]) * 1e3, "k--", lw=0.8, label="commanded")
    ax[0, 0].plot(rs["X"][keep] * 1e3, rs["Y"][keep] * 1e3, color="C3", lw=1.3, label="shaft loop")
    ax[0, 0].plot(rr["X"][keep] * 1e3, rr["Y"][keep] * 1e3, color="C0", lw=1.3, label="string loop")
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("(a) traced circle: shaft loop notches on backlash")
    ax[0, 0].set_xlabel("X  [mm]"); ax[0, 0].set_ylabel("Y  [mm]")
    ax[0, 0].legend(fontsize=8, loc="upper right"); ax[0, 0].grid(alpha=0.3)

    # (b) radial error vs angle around the circle
    def radial(r):
        k = r["t"] > CIRCLE_PERIOD * 0.6
        ang = np.arctan2(r["Y"][k] - cy, r["X"][k] - cx)
        rad = np.hypot(r["X"][k] - cx, r["Y"][k] - cy)
        o = np.argsort(ang)
        return np.degrees(ang[o]), (rad[o] - R) * 1e6
    a1, e1 = radial(rs); a2, e2 = radial(rr)
    ax[0, 1].plot(a1, e1, color="C3", lw=1.0, label="shaft loop")
    ax[0, 1].plot(a2, e2, color="C0", lw=1.0, label="string loop")
    ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[0, 1].set_title("(b) radial error vs angle — notches where a belt reverses")
    ax[0, 1].set_xlabel("angle around circle  [deg]"); ax[0, 1].set_ylabel("radial error  [µm]")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # (c) position-dependent belt stretch: hold across X under a 5 N Y-load
    Xs = np.linspace(cx - 0.06, cx + 0.06, 9)
    dh = hold_droop(pr, "shaft", Xs, cy, load=(0.0, 5.0)) * 1e6
    dr = hold_droop(pr, "string", Xs, cy, load=(0.0, 5.0)) * 1e6
    ax[1, 0].plot(Xs * 1e3, dh, "o-", color="C3", ms=4, label="shaft loop")
    ax[1, 0].plot(Xs * 1e3, dr, "o-", color="C0", ms=4, label="string loop")
    ax[1, 0].set_title("(c) hold under 5 N load: shaft droops (belt stretch), string holds")
    ax[1, 0].set_xlabel("toolhead X  [mm]"); ax[1, 0].set_ylabel("position error  [µm]")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (d) error summary bars: backlash (circle, reuse a's runs) + stretch (load hold)
    def rms(r):
        k = r["t"] > CIRCLE_PERIOD * 0.6
        return np.sqrt(np.mean((np.hypot(r["X"][k] - r["Xc"][k], r["Y"][k] - r["Yc"][k]))**2)) * 1e6
    labels = ["circle\ntrace", "5 N\nload hold"]
    esh = [rms(rs), np.mean(dh)]
    est = [rms(rr), np.mean(dr)]
    xs = np.arange(len(labels)); w = 0.36
    b1 = ax[1, 1].bar(xs - w / 2, esh, w, color="C3", label="shaft loop")
    b2 = ax[1, 1].bar(xs + w / 2, est, w, color="C0", label="string loop")
    for bars in (b1, b2):
        for r in bars:
            ax[1, 1].annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                              ha="center", va="bottom", fontsize=8)
    ax[1, 1].axhline(pr.string_sigma * 1e6, color="C0", ls="--", lw=0.8, label="scale noise")
    ax[1, 1].axhline(pr.backlash * 1e6, color="C3", ls=":", lw=0.8, label="1 backlash")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels(labels, fontsize=9)
    ax[1, 1].set_title("(d) RMS toolhead error — string beats shaft on both")
    ax[1, 1].set_ylabel("error  [µm]")
    ax[1, 1].legend(fontsize=8, loc="center right"); ax[1, 1].grid(alpha=0.3, which="both", axis="y")

    fig.suptitle("CoreXY on cheap TT motors — the loop must close on the toolhead (draw-wire), "
                 "not the motors", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "corexy.png", dpi=120)
    print(f"  wrote {OUT/'corexy.png'}")


if __name__ == "__main__":
    pr = report()
    render(pr)
