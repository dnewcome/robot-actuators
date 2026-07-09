"""
Kinematic MuJoCo viewer for the two-stage compound-differential cycloidal.

Drives every joint at its true rate so the 121:1 reduction is visible at a glance —
the eccentric races, the discs orbit, the output drum barely moves:

  output drum (ring2)      1x        theta            (the slow body output)
  eccentric input          ratio x   ratio*theta      (co-rotating, the fast input)
  disc1 / disc2 (rel ecc)  -ratio*(z1+1)/z1 * theta   (orbit @ input freq, spin slow)
  central disc (coupler)   -ratio/z1 * theta          (about the main axis)

Ratio: u = z1*(z2+1)/(z1-z2)  (compound differential; z1=11,z2=10 -> 121:1).

    ../.venv/bin/python cycloidal-2stage/sim.py          # interactive, 0.20 rev/s out
    ../.venv/bin/python cycloidal-2stage/sim.py 0.05     # slower
    ../.venv/bin/python cycloidal-2stage/sim.py render    # headless -> out/sim.gif + montage
"""

import sys
import time
from math import pi
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive import Params

HERE = Path(__file__).resolve().parent


def _rates(p: Params):
    ratio = p.ratio
    return {
        "joint_out": 1.0,
        "joint_ecc": ratio,
        "joint_disc1": -ratio * (p.z1 + 1) / p.z1,
        "joint_disc2": -ratio * (p.z1 + 1) / p.z1,
        "joint_cd": -ratio / p.z1,
    }


def _apply(m, d, rates, theta):
    for name, w in rates.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = w * theta
    mujoco.mj_forward(m, d)


def view(p: Params, out_rev_per_s: float = 0.20):
    from mujoco import viewer as mjv
    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)
    rates = _rates(p)

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
        print(f"  speed: output {eff:.3f} rev/s  (input {eff*p.ratio:.1f} rev/s)"
              f"{'  [PAUSED]' if sc['paused'] else ''}")

    dt = 1.0 / 60.0
    print(f"\nviewer: {p.ratio:.0f}:1 two-stage cycloidal | output {out_rev_per_s:.2f} rev/s | "
          f"input {out_rev_per_s*p.ratio:.1f} rev/s (co-rotating)")
    print("  controls: up/down faster/slower · SPACE pause · R reset · close window to exit")
    with mjv.launch_passive(m, d, key_callback=on_key) as v:
        theta = 0.0
        while v.is_running():
            if not sc["paused"]:
                theta += out_rev_per_s * sc["mult"] * 2 * pi * dt
            _apply(m, d, rates, theta)
            v.sync()
            time.sleep(dt)


def render(p: Params, sweep_in_revs: float = 3.0, n_frames: int = 60,
           height: int = 460, width: int = 900):
    """Headless: sweep a few INPUT revolutions (so the fast internals whirl while the
    output barely nudges) and write out/sim.gif + a 6-frame montage."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)
    rates = _rates(p)
    try:
        r = mujoco.Renderer(m, height=height, width=width)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        r = mujoco.Renderer(m, height=height, width=width)

    theta_end = sweep_in_revs * 2 * pi / p.ratio      # output angle after that many input revs
    frames = []
    for i in range(n_frames):
        theta = theta_end * i / (n_frames - 1)
        _apply(m, d, rates, theta)
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
    picks = [round(x) for x in
             [0, n_frames / 6, 2 * n_frames / 6, 3 * n_frames / 6,
              4 * n_frames / 6, n_frames - 1]]
    fig, ax = plt.subplots(1, 6, figsize=(16, 3))
    for a, k in zip(ax, picks):
        frac = k / (n_frames - 1)
        a.imshow(frames[k]); a.set_axis_off()
        a.set_title(f"in {sweep_in_revs*360*frac:.0f}° / out {sweep_in_revs*360*frac/p.ratio:.1f}°",
                    fontsize=8)
    fig.suptitle(f"{p.ratio:.0f}:1 two-stage cycloidal — {sweep_in_revs:.0f} input revolutions "
                 f"turn the output just {sweep_in_revs*360/p.ratio:.1f}°", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "out" / "sim_montage.png", dpi=110)
    print(f"  wrote {HERE/'out'/'sim_montage.png'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "render":
        render(Params())
    else:
        view(Params(), out_rev_per_s=float(arg) if arg else 0.20)
