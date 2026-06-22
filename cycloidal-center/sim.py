"""
Kinematic MuJoCo viewer for the straddle-carrier centre-output cycloidal.

Drives every joint at its true cycloidal rate so the reduction is visible:
  output/carrier  1x  (theta)
  eccentric input  -ratio x        (fast, opposite sense)
  disc             orbits at the input frequency (radius E) AND spins with the
                   carrier — its rotation RELATIVE to the eccentric it rides on is
                   theta - theta_ecc = theta*(1 + ratio).

Single-stage cycloidal, ring fixed, carrier out: |ratio| = lobes, output reversed.

    ../.venv/bin/python cycloidal-center/sim.py          # default 0.20 rev/s output
    ../.venv/bin/python cycloidal-center/sim.py 0.05     # slower
"""

import sys
import time
from math import pi
from pathlib import Path

import mujoco
from mujoco import viewer as mjv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive import Params

HERE = Path(__file__).resolve().parent


def view(p: Params, out_rev_per_s: float = 0.20):
    ratio = p.ratio                                   # output -> eccentric (the input)

    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)

    def q(name):
        return m.jnt_qposadr[m.joint(name).id]
    q_ecc, q_disc, q_out = q("joint_ecc"), q("joint_disc"), q("joint_out")

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
        print(f"  speed: output {eff:.3f} rev/s  (input {eff*ratio:.2f} rev/s)"
              f"{'  [PAUSED]' if sc['paused'] else ''}")

    dt = 1.0 / 60.0
    print(f"\nviewer: {ratio:.0f}:1 cycloidal | output {out_rev_per_s:.2f} rev/s | "
          f"input {out_rev_per_s*ratio:.2f} rev/s (reversed) | disc orbit @ input freq")
    print("  controls: up/down faster/slower · SPACE pause · R reset · close window to exit")
    with mjv.launch_passive(m, d, key_callback=on_key) as v:
        theta = 0.0
        while v.is_running():
            if not sc["paused"]:
                theta += out_rev_per_s * sc["mult"] * 2 * pi * dt
            d.qpos[q_out] = theta
            d.qpos[q_ecc] = -ratio * theta
            d.qpos[q_disc] = theta * (1 + ratio)          # relative to the eccentric
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(dt)


if __name__ == "__main__":
    rev = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    view(Params(), out_rev_per_s=rev)
