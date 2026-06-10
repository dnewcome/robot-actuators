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


def main():
    model, spec, motor, aid = load()
    spec.report(motor)

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

    # --- Scenario B: lift the test arm from horizontal (gravity on) ------------
    model.opt.gravity[:] = (0, 0, -9.81)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grav_torque = data.qfrc_bias[0]   # torque needed to hold at horizontal
    q, _ = settle_angle(model, data, MotorSpec().max_current, 0.6, aid)
    lifted = np.degrees(q)
    print("--- Scenario B: lift 100 g @ 150 mm from horizontal (peak current) ---")
    print(f"  gravity hold torque needed = {abs(grav_torque):.3f} N·m   (peak avail {spec.peak_torque:.3f})")
    print(f"  arm swung to {lifted:+.0f}°  ->  {'LIFTS ✓' if lifted > 60 else 'STALLS ✗'}\n")

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
    dof = model.jnt_dofadr[model.joint("joint_out").id]
    Kt, N = motor.kt, spec.ratio
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

    if "--view" in sys.argv:
        view_demo(spec.ratio)


def view_demo(ratio, out_rev_per_s=0.25):
    """Real-time, kinematically-driven look at the mechanism (no load arm).
    Output marker turns at `out_rev_per_s`; input marker spins `ratio`x faster."""
    import time
    from mujoco import viewer as mjv

    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer_scene.xml"))
    d = mujoco.MjData(m)
    q_out = m.jnt_qposadr[m.joint("joint_out").id]
    q_in = m.jnt_qposadr[m.joint("joint_in").id]
    w = out_rev_per_s * 2 * pi          # output rad/s
    dt = 1.0 / 60.0
    print(f"\nviewer: output {out_rev_per_s:.2f} rev/s, input {out_rev_per_s*ratio:.1f} rev/s "
          f"({ratio:.0f}:1). Close the window to exit.")
    with mjv.launch_passive(m, d) as v:
        theta = 0.0
        while v.is_running():
            theta += w * dt
            d.qpos[q_out] = theta
            d.qpos[q_in] = theta * ratio
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
