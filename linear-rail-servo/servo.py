"""
Dynamic control proof: a GARBAGE TT gearmotor becomes a PRECISION linear servo when
the loop is closed on the capacitive rail scale instead of the motor shaft.

This is the point of the whole folder. A yellow TT gearmotor has ~1/4 mm of gearbox
backlash, plastic-gear Coulomb friction, and a soft torque-speed line — nobody would
call it precise. But precision is a property of the *measurement*, not the actuator: if
a ~1 µm capacitive scale (encoder.py) watches the CARRIAGE directly and the controller
closes around it, every bit of backlash and friction is *inside* the loop and gets driven
out. Close the same loop on the motor's own shaft encoder instead and the backlash sits
*outside* the loop — the carriage then lags by up to the full lash and no amount of gain
fixes it. This file simulates both and shows the difference.

MODEL — a faithful-enough 2-mass belt drive with the TT motor's real garbage:

    motor(+gearbox)  --belt(spring)--[ backlash gap ±δ ]--  carriage --friction-- rail
      M_m, torque-speed line,           k_belt, c_belt        M_c, viscous+stiction
      Coulomb friction, cogging

  state (linear-equivalent units, belt position m = r·θ_motor):
    M_m·m̈ = clip(u, ±f_stall) − (f_stall/v_noload)·ṁ − f_coulomb·s(ṁ) − F_couple
    M_c·ẍ = F_couple − c_rail·ẋ − f_stick·s(ẋ) − F_ext
    F_couple = k_belt·deadband(m−x, δ) + c_belt·(ṁ−ẋ)·[engaged]      (backlash: no force
               inside the gap; a compliant belt spring once the teeth re-seat)

  The motor is a linear torque-speed source (force cap f_stall, back-EMF droop
  f_stall/v_noload) — the standard first-order DC-motor line. s(·)=tanh(v/vε) smooths
  Coulomb friction through zero. Hard end-stops clamp x∈[0, travel].

  FEEDBACK MODES:
    rail  — controller reads the carriage via the cap scale (σ ≈ encoder.res_fine, ~1 µm).
            Backlash is INSIDE the loop → cancelled. This is the servo.
    shaft — controller reads the motor shaft (belt position m) via a rotary encoder.
            Backlash is OUTSIDE the loop → carriage lags by ~δ, reversals lose motion.

Trustworthy: the qualitative backlash-cancellation, the reversal lost-motion (≈ 2δ for
the shaft loop, → the rail-loop in-position window for the rail loop), the error-vs-
backlash scaling. First-order: lumped belt, smoothed friction, model-based feedforward,
a mechanics-limited in-position window, no motor electrical dynamics (the back-EMF line
stands in for them). Rail-loop hold is mechanics-limited (~a few µm), NOT scale-limited.

    ../.venv/bin/python linear-rail-servo/servo.py            # report + out/servo.png
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from encoder import CapVernierEncoder

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


@dataclass
class ServoParams:
    # --- load / rail -----------------------------------------------------------
    carriage_mass_kg: float = 0.30
    travel_mm: float = 150.0
    c_rail: float = 8.0              # viscous rail friction [N·s/m]
    f_stick: float = 0.3            # rail Coulomb/stiction [N]

    # --- TT gearmotor (the garbage), referred to the OUTPUT/pulley -------------
    pulley_radius_mm: float = 8.0    # x = r·θ ; belt pitch radius
    gear_ratio: float = 48.0         # TT "48:1" (documentation only; effects folded below)
    out_stall_torque_Nm: float = 0.06    # output stall torque (~0.6 kg·cm)
    out_noload_rpm: float = 200.0        # output no-load speed
    out_inertia: float = 4.0e-5      # output-referred rotary inertia (reflected rotor) [kg·m²]
    f_coulomb: float = 0.1           # gearbox Coulomb friction, referred to belt [N]
    backlash_deg: float = 3.0        # gearbox lash at the output shaft (TT motors: awful)
    cog_frac: float = 0.04           # cogging torque ripple fraction
    cog_teeth: float = 12.0          # cogging periods per output rev

    # --- belt ------------------------------------------------------------------
    belt_stiffness: float = 8.0e4    # coupling spring once teeth are seated [N/m]
    belt_damping: float = 30.0       # [N·s/m]

    # --- controller (PID on position, discrete) --------------------------------
    # Tuned to the ACHIEVABLE plant: the motor makes only f_stall ≈ 6 N against ~1.9 kg of
    # reflected+carriage mass, so useful bandwidth is ω_n ≈ 20 rad/s (~3 Hz), near-critical.
    # Hotter gains just saturate the motor and limit-cycle against the backlash.
    kp: float = 6000.0               # [N/m]   feedback trims; feedforward does the move
    ki: float = 8000.0               # [N/(m·s)]
    kd: float = 150.0                # [N·s/m] (with back-EMF droop → well damped)
    d_filt_hz: float = 30.0          # derivative low-pass (tames encoder-step kicks)
    feedforward: bool = True         # model-based v/a feedforward from the profile
    in_pos_um: float = 8.0           # rail-loop in-position window: inside it the drive
                                     #   relaxes so friction holds (kills integrator hunting).
                                     #   Mechanics-limited, NOT sensor-limited (scale ~1 µm).
    control_hz: float = 1000.0
    u_slew: float = 6.0e4            # command slew limit [N/s] (avoid infinite kicks)
    shaft_enc_res_mm: float = 0.05   # rotary shaft encoder LSB (~576 cpr at the pulley)

    # --- sim -------------------------------------------------------------------
    dt: float = 1.0e-4
    enc: CapVernierEncoder = field(default_factory=CapVernierEncoder)

    # ---- derived (SI) ---------------------------------------------------------
    @property
    def r(self) -> float:
        return self.pulley_radius_mm * 1e-3

    @property
    def travel(self) -> float:
        return self.travel_mm * 1e-3

    @property
    def motor_mass(self) -> float:                 # belt-referred motor inertia [kg]
        return self.out_inertia / self.r ** 2

    @property
    def f_stall(self) -> float:                    # belt-referred stall force [N]
        return self.out_stall_torque_Nm / self.r

    @property
    def v_noload(self) -> float:                   # belt-referred no-load speed [m/s]
        return (self.out_noload_rpm * 2 * np.pi / 60.0) * self.r

    @property
    def backlash(self) -> float:                   # lost motion at the belt [m]
        return np.deg2rad(self.backlash_deg) * self.r

    @property
    def rail_sigma(self) -> float:                 # cap-scale position noise [m]
        return self.enc.res_fine


def _deadband(e, d):
    """Backlash: transmits nothing while |e| < d, else the overshoot past the gap."""
    return np.maximum(0.0, e - d) + np.minimum(0.0, e + d)


def simulate(p: ServoParams, target, mode: str = "rail", t_end: float = 2.5, seed: int = 0):
    """Integrate the 2-mass drive under PID feedback. `target` is a callable t -> x_cmd [m].
    mode: 'rail' (loop on the cap scale) or 'shaft' (loop on the motor encoder)."""
    rng = np.random.default_rng(seed)
    dt = p.dt
    n = int(t_end / dt)
    ctrl_every = max(1, int(round(1.0 / (p.control_hz * dt))))

    # state
    x = float(target(0.0)); xd = 0.0          # carriage
    m = x; md = 0.0                            # motor-side belt position
    integ = 0.0; e_prev = 0.0; deriv = 0.0; u = 0.0
    M_m, M_c = p.motor_mass, p.carriage_mass_kg
    b_emf = p.f_stall / p.v_noload             # torque-speed droop coefficient
    db, k, cb = p.backlash, p.belt_stiffness, p.belt_damping
    veps = 1e-4
    a_d = np.exp(-2 * np.pi * p.d_filt_hz / p.control_hz)   # derivative EMA pole
    i_max = 2 * p.f_stall / p.ki               # anti-windup: integral force ≤ 2·f_stall
    # in-position window: inside it the drive relaxes (integrator bleeds) so friction
    # holds — no integrator/Coulomb hunting. Rail loop is mechanics-limited (backlash +
    # weak motor), NOT sensor-limited; the shaft loop's window is its encoder LSB.
    e_dead = p.in_pos_um * 1e-6 if mode == "rail" else 0.7 * p.shaft_enc_res_mm * 1e-3

    t_arr = np.empty(n); x_arr = np.empty(n); m_arr = np.empty(n)
    tgt_arr = np.empty(n); u_arr = np.empty(n); meas_arr = np.empty(n)
    meas = x

    for i in range(n):
        t = i * dt
        xc = target(t)

        # ---- controller (runs at control_hz, holds between ticks) ----
        if i % ctrl_every == 0:
            if mode == "rail":
                meas = x + rng.normal(0.0, p.rail_sigma)
            else:  # shaft: rotary encoder on the motor, quantized, blind to backlash
                q = p.shaft_enc_res_mm * 1e-3
                meas = np.round(m / q) * q + rng.normal(0.0, q * 0.3)
            e = xc - meas
            deriv = a_d * deriv + (1 - a_d) * (e - e_prev) * p.control_hz  # filtered D
            e_prev = e
            # model-based feedforward from the reference (unburdens the feedback):
            # belt force to accelerate both masses + beat the motor's back-EMF droop
            u_ff = 0.0
            if p.feedforward:
                h = 1.0 / p.control_hz
                v_ref = (target(t + h) - target(t - h)) / (2 * h)
                a_ref = (target(t + h) - 2 * xc + target(t - h)) / (h * h)
                u_ff = (M_m + M_c) * a_ref + b_emf * v_ref
            if abs(e) < e_dead:
                integ *= 0.98                    # in position: bleed drive → friction holds
                u_pd = p.kd * deriv
            else:
                u_pd = p.kp * e + p.kd * deriv
                # conditional anti-windup: wind the integral only when not saturated
                if abs(u_ff + u_pd + p.ki * integ) < p.f_stall:
                    integ = np.clip(integ + e / p.control_hz, -i_max, i_max)
            u_cmd = u_ff + u_pd + p.ki * integ
            du = np.clip(u_cmd - u, -p.u_slew / p.control_hz, p.u_slew / p.control_hz)
            u = u + du

        # ---- motor: torque-speed line + cogging + Coulomb ----
        cog = p.cog_frac * p.f_stall * np.sin(p.cog_teeth * m / p.r)
        f_motor = np.clip(u, -p.f_stall, p.f_stall) - b_emf * md + cog \
            - p.f_coulomb * np.tanh(md / veps)

        # ---- belt coupling through the backlash gap ----
        e_gap = m - x
        spring = k * _deadband(e_gap, db)
        engaged = abs(e_gap) > db
        F_couple = spring + (cb * (md - xd) if engaged else 0.0)

        # ---- accelerations ----
        a_m = (f_motor - F_couple) / M_m
        a_x = (F_couple - p.c_rail * xd - p.f_stick * np.tanh(xd / veps)) / M_c

        # ---- semi-implicit Euler ----
        md += a_m * dt; m += md * dt
        xd += a_x * dt; x += xd * dt

        # hard end-stops
        if x < 0.0:
            x = 0.0; xd = 0.0
        elif x > p.travel:
            x = p.travel; xd = 0.0

        t_arr[i] = t; x_arr[i] = x; m_arr[i] = m
        tgt_arr[i] = xc; u_arr[i] = u; meas_arr[i] = meas

    return dict(t=t_arr, x=x_arr, m=m_arr, tgt=tgt_arr, u=u_arr, meas=meas_arr)


# ---- scenarios --------------------------------------------------------------
def step_target(x0, x1, t_step=0.1):
    return lambda t: x0 if t < t_step else x1


def profile_target(x0, x1, t0=0.1, v_max=0.09, a_max=0.6):
    """Trapezoidal velocity/accel-limited reference — what a real servo commands a weak
    motor to follow (a bare step just saturates it). Respects the TT motor's speed line."""
    s = np.sign(x1 - x0); D = abs(x1 - x0)
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc ** 2
    if 2 * d_acc >= D:                     # triangular (never reaches v_max)
        t_acc = np.sqrt(D / a_max); d_acc = 0.5 * a_max * t_acc ** 2; t_flat = 0.0
        vpk = a_max * t_acc
    else:
        t_flat = (D - 2 * d_acc) / v_max; vpk = v_max
    T = 2 * t_acc + t_flat

    def f(t):
        tau = t - t0
        if tau <= 0:
            return x0
        if tau >= T:
            return x1
        if tau < t_acc:
            d = 0.5 * a_max * tau ** 2
        elif tau < t_acc + t_flat:
            d = d_acc + vpk * (tau - t_acc)
        else:
            td = tau - (t_acc + t_flat)
            d = d_acc + vpk * t_flat + vpk * td - 0.5 * a_max * td ** 2
        return x0 + s * d
    return f


def _profile_dur(D, v_max=0.09, a_max=0.6):
    D = abs(D); t_acc = v_max / a_max; d_acc = 0.5 * a_max * t_acc ** 2
    return 2 * np.sqrt(D / a_max) if 2 * d_acc >= D else 2 * t_acc + (D - 2 * d_acc) / v_max


def staircase_target(levels, dwell=0.45, t0=0.2):
    """A dwell-at-each-step command up then down. Because it fully settles at every level,
    tracking lag drops out and only the backlash hysteresis remains — the clean way to draw
    the command→carriage loop for a slow loop. Returns (target_fn, sample_times, levels)."""
    T = [0.0, t0]; X = [levels[0], levels[0]]; marks = []
    for lv in levels[1:]:
        d = _profile_dur(lv - X[-1])
        T.append(T[-1] + d); X.append(lv)          # ramp to lv
        T.append(T[-1] + dwell); X.append(lv)      # dwell
        marks.append(T[-1])                        # sample at end of dwell
    Ta, Xa = np.array(T), np.array(X)

    def f(t):
        return float(np.interp(t, Ta, Xa))
    return f, marks, T[-1]


def triangle_target(lo, hi, period, t0=0.2):
    def f(t):
        if t < t0:
            return lo
        ph = ((t - t0) / period) % 1.0
        return lo + (hi - lo) * (2 * ph if ph < 0.5 else 2 * (1 - ph))
    return f


def final_error(p, target, mode, t_end):
    r = simulate(p, target, mode, t_end)
    tail = r["x"][int(0.9 * len(r["x"])):]
    xc = r["tgt"][-1]
    return abs(np.mean(tail) - xc)


def settle_at(p, mode, x_from, x_to, settle=1.4):
    """Profile-move to x_to starting from x_from, fully settle, return the mean carriage
    position. Full settle removes tracking lag so only the backlash offset remains."""
    r = simulate(p, profile_target(x_from, x_to, 0.1), mode, t_end=0.1 + settle)
    return np.mean(r["x"][int(0.85 * len(r["x"])):])


def lost_motion(p, mode, mid=0.075, delta=0.015, settle=2.5):
    """Reversal lost motion: settle onto `mid` approached from BELOW vs from ABOVE. For the
    shaft loop the carriage rests ~±backlash either side of the motor target → ≈ full lash;
    for the rail loop the carriage lands on `mid` both ways → just the scale noise."""
    up = settle_at(p, mode, mid - delta, mid, settle)    # arriving while moving forward
    down = settle_at(p, mode, mid + delta, mid, settle)  # arriving while moving backward
    return abs(up - down)


def report(p: ServoParams | None = None):
    p = p or ServoParams()
    print("\n" + "=" * 68)
    print("  TT-MOTOR LINEAR SERVO  —  loop on the rail scale vs the motor shaft")
    print("=" * 68)
    print(f"  carriage {p.carriage_mass_kg*1e3:.0f} g, travel {p.travel_mm:.0f} mm, "
          f"pulley Ø{2*p.pulley_radius_mm:.0f} mm")
    print(f"  motor: stall {p.out_stall_torque_Nm*1e2:.1f} N·cm, {p.out_noload_rpm:.0f} rpm "
          f"no-load  -> belt f_stall {p.f_stall:.1f} N, v_noload {p.v_noload*1e3:.0f} mm/s")
    print(f"  reflected motor mass {p.motor_mass:.2f} kg  (> carriage: 48:1 gear ratio)")
    print(f"  GEARBOX BACKLASH {p.backlash_deg:.1f}° = {p.backlash*1e6:.0f} µm at the belt")
    print(f"  rail scale noise σ = {p.rail_sigma*1e6:.2f} µm   "
          f"shaft encoder LSB = {p.shaft_enc_res_mm*1e3:.0f} µm")
    print("-" * 68)
    tgt = profile_target(0.050, 0.100, 0.1)
    for mode in ("shaft", "rail"):
        r = simulate(p, tgt, mode, t_end=2.0)
        tail = r["x"][int(0.85 * len(r["x"])):]
        err = (np.mean(tail) - 0.100) * 1e6
        ripple = np.std(tail) * 1e6
        lost = lost_motion(p, mode) * 1e6
        print(f"  loop on {mode:>5}:  settle error {err:+7.1f} µm, ripple {ripple:5.1f} µm,"
              f"  reversal lost-motion {lost:6.1f} µm")
    print("-" * 68)
    print(f"  backlash at belt = {p.backlash*1e6:.0f} µm.  Closing on the RAIL removes it "
          f"(lost motion → scale noise);")
    print("  closing on the SHAFT leaves the full lash as reversal dead-band. Same motor.")
    print("=" * 68 + "\n")
    return p


def render(p: ServoParams | None = None):
    p = p or ServoParams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a) profiled 50 -> 100 mm move, both loops
    tgt = profile_target(0.050, 0.100, 0.1)
    rs = simulate(p, tgt, "shaft", 1.6)
    rr = simulate(p, tgt, "rail", 1.6)
    ax[0, 0].plot(rs["t"], rs["tgt"] * 1e3, "k--", lw=0.9, label="profiled command")
    ax[0, 0].plot(rs["t"], rs["x"] * 1e3, color="C3", lw=1.5, label="carriage — shaft loop")
    ax[0, 0].plot(rr["t"], rr["x"] * 1e3, color="C0", lw=1.5, label="carriage — rail loop")
    ax[0, 0].set_title("(a) 50→100 mm profiled move: same motor, two loops")
    ax[0, 0].set_xlabel("time  [s]"); ax[0, 0].set_ylabel("position  [mm]")
    ax[0, 0].legend(fontsize=8, loc="lower right"); ax[0, 0].grid(alpha=0.3)

    # (b) settle zoom near target — the shaft loop parks off by ~the lash
    ax[0, 1].axhline(100.0, color="k", ls="--", lw=0.9, label="command 100 mm")
    ax[0, 1].axhspan(100 - p.backlash * 1e3, 100 + p.backlash * 1e3, color="grey", alpha=0.15,
                     label=f"±backlash ({p.backlash*1e3:.2f} mm)")
    ax[0, 1].plot(rs["t"], rs["x"] * 1e3, color="C3", lw=1.4, label="shaft loop")
    ax[0, 1].plot(rr["t"], rr["x"] * 1e3, color="C0", lw=1.4, label="rail loop")
    ax[0, 1].set_ylim(100 - 2.5 * p.backlash * 1e3, 100 + 2.5 * p.backlash * 1e3)
    ax[0, 1].set_xlim(0.4, 1.6)
    ax[0, 1].set_title("(b) settle zoom: rail loop holds the carriage on target")
    ax[0, 1].set_xlabel("time  [s]"); ax[0, 1].set_ylabel("position  [mm]")
    ax[0, 1].legend(fontsize=8, loc="upper right"); ax[0, 1].grid(alpha=0.3)

    # (c) hysteresis: SETTLED carriage vs command up-then-down (tracking lag removed)
    levels = np.concatenate([np.arange(40, 60.01, 4), np.arange(56, 39.99, -4)]) * 1e-3
    stair, marks, t_end = staircase_target(levels, dwell=0.45)
    ax[1, 0].plot([40, 60], [40, 60], "k:", lw=0.8, label="perfect (y=x)")
    for mode, col, lab in [("shaft", "C3", "shaft loop — lost motion ≈ lash"),
                           ("rail", "C0", "rail loop — lands on command")]:
        r = simulate(p, stair, mode, t_end + 0.1)
        cmd = np.array([stair(t) for t in marks]) * 1e3
        xs = np.array([r["x"][int(t / p.dt)] for t in marks]) * 1e3
        ax[1, 0].plot(cmd, xs, "o-", color=col, ms=4, lw=1.3, label=lab)
    ax[1, 0].set_title("(c) settled carriage vs command: shaft opens a backlash loop")
    ax[1, 0].set_xlabel("commanded  [mm]"); ax[1, 0].set_ylabel("settled carriage  [mm]")
    ax[1, 0].legend(fontsize=8, loc="upper left"); ax[1, 0].grid(alpha=0.3)

    # (d) reversal lost-motion vs backlash — sweep the realistic TT range (up to ~3°)
    lashes = np.linspace(0.3, 3.0, 10)
    es, er = [], []
    for L in lashes:
        q = ServoParams(**{**p.__dict__, "backlash_deg": L})
        es.append(lost_motion(q, "shaft") * 1e6)
        er.append(lost_motion(q, "rail") * 1e6)
    lash_um = np.deg2rad(lashes) * p.r * 1e6
    ax[1, 1].plot(lash_um, es, "o-", color="C3", label="shaft loop")
    ax[1, 1].plot(lash_um, er, "o-", color="C0", label="rail loop")
    ax[1, 1].plot(lash_um, lash_um, "k:", lw=0.8, label="= backlash")
    ax[1, 1].axhline(p.rail_sigma * 1e6, color="C0", ls="--", lw=0.8, label="scale noise floor")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("(d) reversal lost-motion vs gearbox backlash")
    ax[1, 1].set_xlabel("backlash at belt  [µm]"); ax[1, 1].set_ylabel("lost motion  [µm]")
    ax[1, 1].legend(fontsize=8, loc="center right"); ax[1, 1].grid(alpha=0.3, which="both")

    fig.suptitle("Crappy TT gearmotor as a precision linear servo — the loop must close on "
                 "the RAIL, not the shaft", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "servo.png", dpi=120)
    print(f"  wrote {OUT/'servo.png'}")


if __name__ == "__main__":
    p = report()
    render(p)
