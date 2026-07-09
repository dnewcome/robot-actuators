"""
Kinematic MuJoCo viewer for the linear rail servo.

The carriage slides along the rail while the two GT2 pulleys spin at theta = x/r_pulley,
so the belt tracks it. The green capacitive scale runs the length of the rail and the gold
slider PCB rides just above it — the encoder that (in servo.py) closes the loop and cancels
the TT gearmotor's backlash. This viewer just animates the mechanism; the control proof is
servo.py and the sensor physics is encoder.py.

    ../.venv/bin/python linear-rail-servo/sim.py           # interactive sweep
    ../.venv/bin/python linear-rail-servo/sim.py 0.3       # faster sweep (rev-ish rate)
    ../.venv/bin/python linear-rail-servo/sim.py render     # headless -> out/sim.gif + montage
"""

import sys
import time
from math import cos, pi
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail import Params

HERE = Path(__file__).resolve().parent
R_PULLEY = 0.006          # pulley pitch radius [m] (Ø12 body) — theta = x / R_PULLEY


def _apply(m, d, x):
    """Place the carriage at x [m] along travel and spin both pulleys to match the belt."""
    d.qpos[m.jnt_qposadr[m.joint("joint_carriage").id]] = x
    theta = x / R_PULLEY
    d.qpos[m.jnt_qposadr[m.joint("joint_drive").id]] = theta
    d.qpos[m.jnt_qposadr[m.joint("joint_idler").id]] = theta
    mujoco.mj_forward(m, d)


def _sweep(travel, phase):
    """Smooth back-and-forth over [0, travel] (a raised-cosine there-and-back)."""
    tri = 1 - abs((phase % 2.0) - 1.0)           # 0->1->0 triangle
    return travel * 0.5 * (1 - cos(pi * tri))


def view(p: Params, cycles_per_s: float = 0.2):
    from mujoco import viewer as mjv
    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)
    travel = p.travel * 1e-3
    dt = 1.0 / 60.0
    print(f"\nviewer: {p.travel:.0f} mm rail servo | carriage sweeps 0..{p.travel:.0f} mm")
    print("  controls: close window to exit")
    with mjv.launch_passive(m, d) as v:
        phase = 0.0
        while v.is_running():
            phase += 2 * cycles_per_s * dt
            _apply(m, d, _sweep(travel, phase))
            v.sync()
            time.sleep(dt)


def render(p: Params, n_frames: int = 60, height: int = 520, width: int = 1100):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)
    travel = p.travel * 1e-3
    try:
        r = mujoco.Renderer(m, height=height, width=width)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        r = mujoco.Renderer(m, height=height, width=width)

    frames = []
    for i in range(n_frames):
        phase = 2.0 * i / (n_frames - 1)          # one full there-and-back
        _apply(m, d, _sweep(travel, phase))
        r.update_scene(d, camera="iso")
        frames.append(r.render().copy())

    (HERE / "out").mkdir(exist_ok=True)
    gif = HERE / "out" / "sim.gif"
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif, frames, fps=20, loop=0)
        print(f"  wrote {gif}")
    except Exception:
        try:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=50, loop=0)
            print(f"  wrote {gif}  (via PIL)")
        except Exception as e:
            print(f"  (gif skipped: {e})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    fig, ax = plt.subplots(len(picks), 1, figsize=(11, 2.1 * len(picks)))
    for a, k in zip(ax, picks):
        x_mm = _sweep(travel, 2.0 * k / (n_frames - 1)) * 1e3
        a.imshow(frames[k]); a.set_axis_off()
        a.set_title(f"carriage at {x_mm:5.0f} mm", fontsize=9, loc="left")
    fig.suptitle(f"{p.travel:.0f} mm capacitive-scale belt axis — TT gearmotor drive "
                 f"({p.enc.res_fine*1e6:.1f} µm scale, absolute over {p.enc.beat_len*1e3:.0f} mm)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(HERE / "out" / "sim_montage.png", dpi=110)
    print(f"  wrote {HERE/'out'/'sim_montage.png'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "render":
        render(Params())
    else:
        view(Params(), cycles_per_s=float(arg) if arg else 0.2)
