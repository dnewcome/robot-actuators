"""
Kinematic MuJoCo viewer for the INVERTED planetary (ring fixed, carrier shaft OUT).

Drives every joint at its true ratio so the 5:1 single stage is visible:
  output/carrier 1x | sun = ratio x (the NEMA-17 stepper INPUT, same sense) |
  planets orbit 1x with the carrier and spin (planet_abs - carrier) about their rollers.

Ring-fixed planetary spin kinematics (referenced to the OUTPUT/carrier = 1x):
  w_planet_abs = w_carrier - (z_sun/z_planet) * (w_sun - w_carrier)
With z_sun=12, z_planet=18, w_sun=ratio=5:  w_planet_abs = 1 - (12/18)*(5-1) = -1.667,
so planets spin ~-2.67x RELATIVE to the carrier they ride on.

    ../.venv/bin/python planetary-inverted/sim.py          # default 0.20 rev/s output
    ../.venv/bin/python planetary-inverted/sim.py 0.05     # slower
"""

import sys
import time
from math import pi
from pathlib import Path

import mujoco
from mujoco import viewer as mjv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive import InvParams

HERE = Path(__file__).resolve().parent


def view(p: InvParams, out_rev_per_s: float = 0.20):
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
    print(f"\nviewer: {ratio:.0f}:1 inverted single stage | output {out_rev_per_s:.2f} rev/s | "
          f"sun {out_rev_per_s*ratio:.2f} rev/s (same sense) | planet spin {planet_rel:+.2f}x")
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
    view(InvParams(), out_rev_per_s=rev)
