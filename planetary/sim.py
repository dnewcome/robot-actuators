"""
Kinematic MuJoCo viewer for the standalone planetary reducer.

Drives every joint at its true ratio so the 4:1 single stage is visible:
  output/carrier 1x | sun = ratio x (the INPUT) | planets orbit 1x with the
  carrier and spin (planet_abs - carrier) about their pins.

Ring-fixed planetary spin kinematics (referenced to the OUTPUT/carrier = 1x):
  w_planet_abs = w_carrier - (z_sun/z_planet) * (w_sun - w_carrier)
With z_sun=z_planet=12 and w_sun=ratio=4:  w_planet_abs = 1 - 1*(4-1) = -2,
so planets spin -3x RELATIVE to the carrier they ride on.

This is the kinematic twin of mujoco/run.py --view, but standalone (no compound
deps). Speed is live: up/down scale it, SPACE pauses, R resets.

    .venv/bin/python planetary/sim.py            # default 0.20 rev/s output
    .venv/bin/python planetary/sim.py 0.05       # slower
"""

import sys
import time
from math import pi
from pathlib import Path

import mujoco
from mujoco import viewer as mjv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reducer import ReducerParams

HERE = Path(__file__).resolve().parent


def view(p: ReducerParams, out_rev_per_s: float = 0.20):
    ratio = p.ratio                                  # output -> sun (the input)
    wc, ws = 1.0, ratio                              # carrier=output=1x, sun=ratio
    w_planet_abs = wc - (p.n_sun / p.n_planet) * (ws - wc)
    planet_rel = w_planet_abs - wc                   # planet spin relative to its carrier

    m = mujoco.MjModel.from_xml_path(str(HERE / "viewer.xml"))
    d = mujoco.MjData(m)

    def q(name):
        return m.jnt_qposadr[m.joint(name).id]
    q_out, q_sun = q("joint_out"), q("joint_sun")
    q_planets = [q(f"joint_planet{i}") for i in range(p.n_planets)]

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
        print(f"  speed: output {eff:.3f} rev/s  (sun {eff*ratio:.2f} rev/s)"
              f"{'  [PAUSED]' if sc['paused'] else ''}")

    dt = 1.0 / 60.0
    print(f"\nviewer: {ratio:.0f}:1 single stage | output {out_rev_per_s:.2f} rev/s | "
          f"sun {out_rev_per_s*ratio:.2f} rev/s | planet spin {planet_rel:+.0f}x")
    print("  controls: up/down faster/slower · SPACE pause · R reset · close window to exit")
    with mjv.launch_passive(m, d, key_callback=on_key) as v:
        theta = 0.0
        while v.is_running():
            if not sc["paused"]:
                theta += out_rev_per_s * sc["mult"] * 2 * pi * dt
            d.qpos[q_out] = theta
            d.qpos[q_sun] = theta * ratio
            for qp in q_planets:
                d.qpos[qp] = theta * planet_rel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(dt)


if __name__ == "__main__":
    rev = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    view(ReducerParams(), out_rev_per_s=rev)
