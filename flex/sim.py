"""
Kinematic MuJoCo viewer for the 2-DOF tendon-driven gimbal.

The platform sweeps a tilt cone; the cross-trunnion gimbal (X-hinge ring + Y-hinge platform)
follows, and the 3 capstan spools spin by the tendon-length change ΔL_i / r_cap computed from
flex.py — so the tendons visibly wind/unwind as the platform leans toward each one in turn.

    ../.venv/bin/python flex/sim.py            # interactive cone sweep
    ../.venv/bin/python flex/sim.py 0.4        # faster
    ../.venv/bin/python flex/sim.py render      # headless -> out/sim.gif + montage
"""

import sys
import time
from math import cos, sin, pi
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flex import FlexParams

HERE = Path(__file__).resolve().parent
P = FlexParams()
TILT = np.deg2rad(20.0)          # cone half-angle to sweep (within the workspace)


def _set(m, d, name, val):
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def _apply(m, d, alpha, beta=TILT):
    ux, uy = beta * cos(alpha), beta * sin(alpha)
    # cross-trunnion gimbal: lean toward (ux,uy) ≈ tilt about Y by ux, about X by −uy
    _set(m, d, "joint_x", -uy)
    _set(m, d, "joint_y", ux)
    # spin each capstan by the tendon it takes up: Δθ = −(L_i − L0_i)/r_cap
    L = P.tendon_lengths(ux, uy)
    for i in range(P.n_tendons):
        _set(m, d, f"joint_cap{i}", -(L[i] - P.L0[i]) / P.r_cap)
    mujoco.mj_forward(m, d)


def view(p_unused=None, cycles_per_s=0.25):
    from mujoco import viewer as mjv
    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)
    dt = 1.0 / 60.0
    print(f"\nviewer: 2-DOF tendon gimbal | platform sweeps a {np.rad2deg(TILT):.0f}° cone")
    print("  close window to exit")
    with mjv.launch_passive(m, d) as v:
        ph = 0.0
        while v.is_running():
            ph += 2 * pi * cycles_per_s * dt
            _apply(m, d, ph)
            v.sync()
            time.sleep(dt)


def render(n_frames=60, height=720, width=960):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)
    try:
        r = mujoco.Renderer(m, height=height, width=width)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        r = mujoco.Renderer(m, height=height, width=width)

    frames = []
    for i in range(n_frames):
        _apply(m, d, 2 * pi * i / (n_frames - 1))
        r.update_scene(d, camera="iso")
        frames.append(r.render().copy())

    (HERE / "out").mkdir(exist_ok=True)
    gif = HERE / "out" / "sim.gif"
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif, frames, fps=20, loop=0)
        print(f"  wrote {gif}")
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print(f"  wrote {gif}  (via PIL)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [0, n_frames // 6, n_frames // 3, n_frames // 2, 2 * n_frames // 3, 5 * n_frames // 6]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for a, k in zip(ax.ravel(), picks):
        alpha = 360 * k / (n_frames - 1)
        a.imshow(frames[k]); a.set_axis_off()
        a.set_title(f"lean toward {alpha:.0f}°", fontsize=9)
    fig.suptitle(f"2-DOF tendon-driven gimbal — 3 capstans sweep the platform through a "
                 f"{np.rad2deg(TILT):.0f}° cone", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "out" / "sim_montage.png", dpi=110)
    print(f"  wrote {HERE/'out'/'sim_montage.png'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "render":
        render()
    else:
        view(cycles_per_s=float(arg) if arg else 0.25)
