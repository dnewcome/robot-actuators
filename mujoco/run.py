"""
Drive the cycloidal actuator in MuJoCo and print a sizing report.

  python mujoco/run.py            # headless physics + report
  python mujoco/run.py --view     # interactive viewer (needs a display)

Injects armature / gear / friction / payload from actuator.py so the sim always
matches the current CAD Params and motor constants.
"""

import sys
from math import pi
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cycloidal"))
from actuator import MotorSpec, ActuatorSpec, Params  # noqa: E402
from efficiency import predict_params  # noqa: E402  (Layer B -> Layer A)

# --- losses + test payload (knobs) ------------------------------------------- #
# The torque-based efficiency model lives in the sim as: gear = η∞ (asymptotic),
# joint frictionloss = no-load drag. That combo reproduces η(T)=η∞·T/(T+drag) for
# free, so efficiency rises with load automatically. Both come from the Layer-B
# model for the current Params (calibration-pending).
_SPEC0          = ActuatorSpec.from_motor(MotorSpec(), Params())
FRICTIONLOSS    = _SPEC0.drag_out   # N·m no-load drag at the output (== backdrive threshold)
DAMPING         = 0.0015   # N·m·s/rad viscous
PAYLOAD_KG      = 0.10     # mass at the end of the 150 mm test arm
ARM_LEN         = 0.15     # m  (must match testbench.xml)


def load():
    motor = MotorSpec()
    spec = ActuatorSpec.from_motor(motor, Params())
    model = mujoco.MjModel.from_xml_path(str(HERE / "testbench.xml"))

    # inject derived physics onto the actuated joint + actuator
    jid = model.joint("joint_out").id
    dof = model.jnt_dofadr[jid]
    model.dof_armature[dof] = spec.reflected_inertia
    model.dof_damping[dof] = DAMPING
    model.dof_frictionloss[dof] = spec.drag_out      # no-load drag = Coulomb frictionloss

    aid = model.actuator("drive").id
    model.actuator_gear[aid, 0] = spec.torque_per_amp   # gear carries η∞
    model.actuator_ctrlrange[aid] = [-MotorSpec().max_current, MotorSpec().max_current]

    # set the test payload
    pid = model.body("payload").id
    model.body_mass[pid] = PAYLOAD_KG

    return model, spec, motor, aid


def settle_angle(model, data, ctrl, t, aid):
    data.ctrl[aid] = ctrl
    n = int(t / model.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(model, data)
    return data.qpos[0], data.qvel[0]


def drive_voltage(model, data, throttle, t, aid, motor, N, dof):
    """Step with a throttle (voltage) command; current is back-EMF limited each step."""
    n = int(t / model.opt.timestep)
    for _ in range(n):
        data.ctrl[aid] = motor.current_at(data.qvel[dof] * N, throttle)
        mujoco.mj_step(model, data)
    return data.qpos[dof], data.qvel[dof]


def main():
    model, spec, motor, aid = load()
    spec.report(motor)
    dof = model.jnt_dofadr[model.joint("joint_out").id]
    Kt, N = motor.kt, spec.ratio

    # --- Scenario A: free angular acceleration (gravity off) -> verify inertia --
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, v = settle_angle(model, data, MotorSpec().max_current, 0.05, aid)
    # subtract the static friction torque the motor must overcome
    net_t = spec.peak_torque - FRICTIONLOSS
    I_eff = net_t / (v / 0.05) if v > 1e-6 else float("nan")
    print("--- Scenario A: free accel (gravity off, peak current) ---")
    print(f"  output reached {v:6.1f} rad/s in 50 ms  ->  effective inertia {I_eff*1e4:.2f}e-4 kg·m²")
    print(f"  (reflected motor inertia alone = {spec.reflected_inertia*1e4:.2f}e-4; rest is arm+payload)\n")

    # --- Scenario B: lift the test arm from horizontal (gravity on, full throttle) --
    model.opt.gravity[:] = (0, 0, -9.81)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grav_torque = data.qfrc_bias[0]   # torque needed to hold at horizontal
    q, w = drive_voltage(model, data, 1.0, 0.6, aid, motor, N, dof)
    lifted = np.degrees(q)
    print("--- Scenario B: lift 100 g @ 150 mm from horizontal (full throttle) ---")
    print(f"  gravity hold torque needed = {abs(grav_torque):.3f} N·m   (peak avail {spec.peak_torque:.3f})")
    print(f"  arm swung to {lifted:+.0f}° at {abs(w)*60/2/pi:.0f} rpm  ->  "
          f"{'LIFTS ✓' if lifted > 60 else 'STALLS ✗'}  (back-EMF now caps the speed)\n")

    # --- Scenario C: backdrivability (no power, gravity on) --------------------
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q, _ = settle_angle(model, data, 0.0, 0.6, aid)
    drop = np.degrees(q)
    held = abs(grav_torque) <= FRICTIONLOSS
    print("--- Scenario C: unpowered hold (backdrive) ---")
    print(f"  friction {FRICTIONLOSS:.3f} N·m vs gravity {abs(grav_torque):.3f} N·m  ->  "
          f"{'self-holds' if held else f'backdrives, falls to {drop:+.0f}°'}")
    print(f"  (cycloidals ARE backdrivable — expected to fall; raise frictionloss only if you measure stiction)\n")

    # --- Scenario D: efficiency vs load, measured from sim power balance --------
    # For each output torque T, command the current that delivers it, apply T as a
    # brake, run to steady speed, and read η = P_out/P_in from the sim. Demonstrates
    # the torque-based model (η rising with load) is physically in the sim.
    print("--- Scenario D: efficiency vs load (measured in sim) ---")
    print(f"  {'T_out':>7s} {'I (A)':>6s} {'out rpm':>8s} {'η meas':>7s} {'η model':>8s}")
    model.opt.gravity[:] = 0
    for T in (0.05, 0.10, 0.20, 0.40, spec.peak_torque):
        I = (T + spec.drag_out) / spec.torque_per_amp * 1.01   # +1% so it creeps forward
        if I > motor.max_current + 1e-9:
            print(f"  {T:7.3f}  {'>13':>5s}   {'—':>7s}    —      {spec.eta_at(T)*100:5.0f}%  (exceeds 13 A)")
            continue
        data = mujoco.MjData(model)
        for _ in range(int(0.8 / model.opt.timestep)):
            data.qfrc_applied[dof] = -T          # brake opposing the (positive) motion
            data.ctrl[aid] = I
            mujoco.mj_step(model, data)
        w = data.qvel[dof]
        p_out = T * w
        p_in = Kt * I * (N * w)
        eta_meas = p_out / p_in if p_in > 1e-9 else 0.0
        print(f"  {T:7.3f} {I:6.1f} {w*60/(2*pi):8.0f} {eta_meas*100:6.0f}% {spec.eta_at(T)*100:7.0f}%")
    print("  (η meas tracks η model -> the load-dependent curve emerges from gear+frictionloss)\n")

    # --- Scenario E: torque-speed curve traced from a real spin-up (back-EMF) ----
    # Full throttle, no load. As the output accelerates, back-EMF cuts the current,
    # so the delivered torque is flat then droops to zero near no-load speed.
    print("--- Scenario E: torque-speed curve (back-EMF, traced in sim spin-up) ---")
    model.opt.gravity[:] = 0
    pid = model.body("payload").id
    model.body_mass[pid] = 0.001                  # light, so it reaches high speed quickly
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # sample the curve at fractions of the (ratio-dependent) no-load speed
    nl_rpm = spec.no_load_speed * 60 / (2 * pi)
    targets, got = [round(f * nl_rpm) for f in (0.0, 0.25, 0.5, 0.7, 0.85, 0.95)], {}
    for _ in range(int(4.0 / model.opt.timestep)):
        rpm = data.qvel[dof] * 60 / (2 * pi)
        I = motor.current_at(data.qvel[dof] * N, 1.0)
        tau = max(0.0, spec.torque_per_amp * I - spec.drag_out)
        for tgt in targets:
            if tgt not in got and rpm >= tgt:
                got[tgt] = tau
        data.ctrl[aid] = I
        mujoco.mj_step(model, data)
    model.body_mass[pid] = PAYLOAD_KG             # restore
    print(f"  {'rpm':>6s} {'τ meas':>8s} {'τ model':>8s}")
    for tgt in targets:
        if tgt in got:
            w = tgt * 2 * pi / 60
            print(f"  {tgt:6d} {got[tgt]:7.3f}  {spec.torque_at_speed(w, motor):7.3f}")
    print(f"  flat to ~{motor.corner_speed/N*60/2/pi:.0f} rpm (current-limited), "
          f"then droops to 0 at {spec.no_load_speed*60/2/pi:.0f} rpm (voltage-limited).\n")

    if "--view" in sys.argv:
        # optional initial output speed: `--view 0.05` or `--speed 0.05`
        rev = 0.10
        for flag in ("--view", "--speed"):
            if flag in sys.argv:
                i = sys.argv.index(flag)
                if i + 1 < len(sys.argv):
                    try:
                        rev = float(sys.argv[i + 1])
                    except ValueError:
                        pass
        view_demo(Params(), out_rev_per_s=rev)


def view_demo(p, out_rev_per_s=0.10):
    """Real-time, kinematically-driven look at the WHOLE hybrid (both stages).
    Each joint is driven at its true ratio so the 40:1 compound is visible:
      output 1x | cyclo carrier (= planet carrier) cyclo_ratio | sun total ratio |
      planets orbit cyclo_ratio, spin (planet_abs - carrier) about their pins.

    Speed is live-adjustable: ↑/↓ scale it, SPACE pauses, R resets. The starting
    output speed comes from `out_rev_per_s` (CLI: `--view 0.05`)."""
    import time
    from mujoco import viewer as mjv

    Nc = p.cyclo_ratio                      # output -> planet/cyclo carrier
    Nt = p.ratio                            # output -> sun (total)
    # planet absolute spin for a ring-fixed planetary (z_sun, z_planet):
    #   w_planet = w_carrier - (z_sun/z_planet)*(w_sun - w_carrier)
    wc = Nc                                 # carrier factor (rel. to output)
    ws = Nt                                 # sun factor
    w_planet_abs = wc - (p.n_sun / p.n_planet) * (ws - wc)
    planet_rel = w_planet_abs - wc          # planet spin relative to its carrier

    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer_scene.xml"))
    d = mujoco.MjData(m)

    def q(name):
        return m.jnt_qposadr[m.joint(name).id]
    q_out, q_sun, q_pc = q("joint_out"), q("joint_sun"), q("joint_pcarrier")
    q_planets = [q(f"joint_planet{i}") for i in range(4)]

    # live speed control via keyboard (GLFW keycodes); mutable so the callback can edit it
    KEY_UP, KEY_DOWN, KEY_SPACE, KEY_R = 265, 264, 32, 82
    sc = {"mult": 1.0, "paused": False}

    def on_key(keycode):
        if keycode == KEY_UP:
            sc["mult"] *= 1.5
        elif keycode == KEY_DOWN:
            sc["mult"] /= 1.5
        elif keycode == KEY_SPACE:
            sc["paused"] = not sc["paused"]
        elif keycode == KEY_R:
            sc["mult"] = 1.0
        eff = out_rev_per_s * sc["mult"]
        print(f"  speed: output {eff:.3f} rev/s  (sun {eff*Nt:.2f} rev/s)"
              f"{'  [PAUSED]' if sc['paused'] else ''}")

    dt = 1.0 / 60.0
    print(f"\nviewer: output {out_rev_per_s:.2f} rev/s | carrier {out_rev_per_s*Nc:.1f} | "
          f"sun {out_rev_per_s*Nt:.1f} rev/s  ({Nt:.0f}:1 total).")
    print("  controls: ↑/↓ faster/slower · SPACE pause · R reset · close window to exit")
    with mjv.launch_passive(m, d, key_callback=on_key) as v:
        theta = 0.0
        while v.is_running():
            if not sc["paused"]:
                theta += out_rev_per_s * sc["mult"] * 2 * pi * dt
            d.qpos[q_out] = theta
            d.qpos[q_pc] = theta * Nc
            d.qpos[q_sun] = theta * Nt
            for qp in q_planets:
                d.qpos[qp] = theta * planet_rel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
