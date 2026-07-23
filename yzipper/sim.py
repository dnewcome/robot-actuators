"""
Kinematic MuJoCo viewer for the Y-ZIPPER — the zipped rod taking a PROGRAMMED curve.

Corrected from the earlier "three strips fold shut" demo. Here the closed rod is built by
instancing the triangular-prism segment mesh along a centreline computed from the paper's
primitive equations (yzipper.inc_straight/inc_bend/inc_coil, Appendix A.3). The slider
travels base→tip; segments behind it read as RIGID (zipped), ahead of it as PENDING — so
you watch the programmed shape (a bend arch, or a coil) rigidize as the zip front advances.

    ../.venv/bin/python yzipper/sim.py             # zip a programmed BEND arch
    ../.venv/bin/python yzipper/sim.py coil        # zip a COIL
    ../.venv/bin/python yzipper/sim.py render      # headless -> out/sim.gif + montage
    ../.venv/bin/python yzipper/sim.py render coil
"""

import sys
import time
from math import pi, sqrt, radians
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yzipper import (YZipperParams, _integrate, inc_straight, inc_bend, inc_coil)
from cad import Cad

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
P = YZipperParams()
C = Cad(P)
m_ = 1e-3


def _program(kind: str):
    """Return centreline frames (meters) for a programmed shape, ~one segment per pitch."""
    p = C.pitch_mm
    if kind == "coil":
        inc = inc_coil(R=16, pitch_h=26, turns=2.2, n=70)
    else:  # bend arch: a short straight lead-in, then a 120° bend
        th = radians(120)
        R = 32.0
        inc = inc_straight(3 * p, 3) + inc_bend(th, R, 40)
    frames = _integrate(inc)
    for F in frames:
        F[:3, 3] *= m_                       # mm -> m
    return frames


def _quat_wxyz(R):
    """Rotation matrix -> (w,x,y,z) quaternion (MuJoCo order)."""
    t = np.trace(R)
    if t > 0:
        s = sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return (w, x, y, z)


def build_xml(kind: str):
    frames = _program(kind)
    n = len(frames) - 1
    segs = ""
    for i in range(n):
        F = frames[i]
        px, py, pz = F[:3, 3]
        w, x, y, z = _quat_wxyz(F[:3, :3])
        segs += (f'<geom name="seg{i}" type="mesh" mesh="segment" material="pending" '
                 f'pos="{px:.5f} {py:.5f} {pz:.5f}" quat="{w:.5f} {x:.5f} {y:.5f} {z:.5f}" '
                 f'contype="0" conaffinity="0"/>')

    xml = f"""<mujoco model="yzipper_program">
  <compiler meshdir="out" angle="radian" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="960" offheight="720"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <mesh name="segment" file="segment.stl" scale="0.001 0.001 0.001"/>
    <mesh name="slider"  file="slider.stl"  scale="0.001 0.001 0.001"/>
    <material name="rigid"   rgba="0.30 0.70 0.40 1"/>
    <material name="pending" rgba="0.75 0.78 0.82 0.35"/>
    <material name="slider"  rgba="0.85 0.68 0.20 1"/>
  </asset>
  <worldbody>
    <light pos="0.12 0.2 0.28" dir="-0.2 -0.4 -1"/>
    <camera name="iso" pos="0.20 0.20 0.15" xyaxes="-0.728 0.686 0 -0.130 -0.138 0.981"/>
    {segs}
    <body name="slider" pos="0 0 0">
      <freejoint name="slidepose"/>
      <geom type="mesh" mesh="slider" material="slider" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""
    (HERE / "viewer.xml").write_text(xml)
    return HERE / "viewer.xml", frames, n


def _apply(m, d, frames, n, zip01):
    """Advance the zip front; recolour segments rigid(behind)/pending(ahead); pose slider."""
    front = int(round(np.clip(zip01, 0, 1) * (n - 1)))
    rigid = np.array([0.30, 0.70, 0.40, 1.0])
    pending = np.array([0.75, 0.78, 0.82, 0.35])
    for i in range(n):
        gid = m.geom(f"seg{i}").id
        m.geom_rgba[gid] = rigid if i <= front else pending
    F = frames[front]
    w, x, y, z = _quat_wxyz(F[:3, :3])
    adr = m.jnt_qposadr[m.joint("slidepose").id]
    d.qpos[adr:adr + 3] = F[:3, 3]
    d.qpos[adr + 3:adr + 7] = (w, x, y, z)
    mujoco.mj_forward(m, d)


def view(kind="bend", cycles_per_s=0.15):
    from mujoco import viewer as mjv
    path, frames, n = build_xml(kind)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    dt = 1.0 / 60.0
    print(f"\nviewer: Y-zipper programmed {kind.upper()} | {n} segments zip base→tip "
          f"(closed rod follows the paper-equation centreline)")
    print("  close window to exit")
    with mjv.launch_passive(m, d) as v:
        ph = 0.0
        while v.is_running():
            ph += 2 * pi * cycles_per_s * dt
            _apply(m, d, frames, n, 0.5 * (1 - np.cos(ph)))
            v.sync()
            time.sleep(dt)


def render(kind="bend", n_frames=48, height=720, width=960):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    path, frames, n = build_xml(kind)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    try:
        r = mujoco.Renderer(m, height=height, width=width)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        r = mujoco.Renderer(m, height=height, width=width)

    imgs, zips = [], []
    for i in range(n_frames):
        z = 0.5 * (1 - np.cos(2 * pi * i / (n_frames - 1)))
        _apply(m, d, frames, n, z)
        zips.append(z)
        r.update_scene(d, camera="iso")
        imgs.append(r.render().copy())

    OUT.mkdir(exist_ok=True)
    gif = OUT / "sim.gif"
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif, imgs, fps=18, loop=0)
        print(f"  wrote {gif}")
    except Exception:
        from PIL import Image
        pi_ = [Image.fromarray(f) for f in imgs]
        pi_[0].save(gif, save_all=True, append_images=pi_[1:], duration=55, loop=0)
        print(f"  wrote {gif}  (via PIL)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [n_frames // 8, n_frames // 4, 3 * n_frames // 8, n_frames // 2,
             5 * n_frames // 8, 3 * n_frames // 4]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for a, k in zip(ax.ravel(), picks):
        a.imshow(imgs[k]); a.set_axis_off()
        a.set_title(f"zip {zips[k]*100:.0f}%  ({int(round(zips[k]*(n-1)))}/{n} segments rigid)", fontsize=9)
    fig.suptitle(f"Y-zipper — closed rod follows the programmed {kind.upper()} centreline "
                 "(paper primitive eqns); slider zips base→tip", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "sim_montage.png", dpi=110)
    print(f"  wrote {OUT/'sim_montage.png'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    do_render = "render" in args
    kind = "coil" if "coil" in args else "bend"
    if do_render:
        render(kind)
    else:
        view(kind)
