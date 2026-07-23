"""
Kinematic MuJoCo viewer for the ZIP-CHAIN LINEAR ACTUATOR.

Rigid parts load as the exported CAD meshes: a static guide HEAD, a SPROCKET that
spins, and the two interlaced STRANDS forming the deployed column on a slide joint.
The sprocket angle and the column extension are GEARED (x = r_p·θ), so spinning the
sprocket deploys/retracts the rigid column out of the head — the mechanism at a glance.
The viewer generates its own model (zipchain/viewer.xml) each run so the geometry
matches params.

The headless montage annotates each frame with the stroke AND the push capacity at
that extension from zipchain.py — so you watch the buckling-limited push ceiling fall
as the column extends (the whole point of the model).

    ../.venv/bin/python zipchain/sim.py            # interactive deploy/retract sweep
    ../.venv/bin/python zipchain/sim.py 0.3        # faster
    ../.venv/bin/python zipchain/sim.py render      # headless -> out/sim.gif + montage
"""

import sys
import time
from math import pi
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zipchain import ZipChainParams
from cad import Cad

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
P = ZipChainParams()
C = Cad(P)

m_ = 1e-3
STROKE = P.stroke_mm * m_
TOTAL = C.total_len * m_
VIS = 2.0 * TOTAL                    # stacked column length so the base stays fed in the head
HEAD = C.head_len * m_
RP = P.r_pitch_m
HALF_PITCH = P.pitch_mm / 2.0 * m_


def build_xml():
    """Generate the viewer model (meshes + geared slide/spin) and write viewer.xml."""
    xml = f"""<mujoco model="zipchain_viewer">
  <compiler meshdir="out" angle="radian" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="960" offheight="720"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <mesh name="head"     file="head.stl"     scale="0.001 0.001 0.001"/>
    <mesh name="sprocket" file="sprocket.stl" scale="0.001 0.001 0.001"/>
    <mesh name="strand"   file="strand.stl"   scale="0.001 0.001 0.001"/>
    <material name="pla"    rgba="0.32 0.35 0.40 1"/>
    <material name="gold"   rgba="0.85 0.68 0.20 1"/>
    <material name="stA"    rgba="0.90 0.90 0.92 1"/>
    <material name="stB"    rgba="0.35 0.72 0.90 1"/>
    <material name="load"   rgba="0.85 0.30 0.25 1"/>
  </asset>

  <worldbody>
    <light pos="0.25 0.5 0.7" dir="-0.2 -0.4 -1"/>
    <camera name="iso" pos="0.52 0.52 0.40" xyaxes="-0.728 0.686 0 -0.130 -0.138 0.981"/>

    <!-- static guide / merge head -->
    <geom type="mesh" mesh="head" material="pla" contype="0" conaffinity="0"/>

    <!-- zip sprocket: spins about Y at the head centre -->
    <body name="sprk" pos="0 0 {HEAD/2:.5f}">
      <joint name="spin" type="hinge" axis="0 1 0"/>
      <geom type="mesh" mesh="sprocket" material="gold" contype="0" conaffinity="0"/>
    </body>

    <!-- deployed column: two interlaced strands on a slide joint (deploy along Z).
         The column is stacked to length VIS = 2·TOTAL so its base stays fed through
         the head across the full stroke instead of lifting off (kinematic viewer). -->
    <body name="column" pos="0 0 {HEAD-VIS:.5f}">
      <joint name="deploy" type="slide" axis="0 0 1" range="0 {STROKE:.5f}"/>
      <geom type="mesh" mesh="strand" material="stA" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="strand" material="stA" contype="0" conaffinity="0"
            pos="0 0 {TOTAL:.5f}"/>
      <geom type="mesh" mesh="strand" material="stB" contype="0" conaffinity="0"
            euler="0 0 3.14159" pos="0 0 {HALF_PITCH:.5f}"/>
      <geom type="mesh" mesh="strand" material="stB" contype="0" conaffinity="0"
            euler="0 0 3.14159" pos="0 0 {TOTAL+HALF_PITCH:.5f}"/>
      <!-- payload riding the column tip -->
      <geom type="box" size="{C.col_w/2*m_*1.6:.5f} {C.col_h/2*m_*1.6:.5f} 0.010"
            pos="0 0 {VIS+0.010:.5f}" material="load" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""
    (HERE / "viewer.xml").write_text(xml)
    return HERE / "viewer.xml"


def _set(m, d, name, val):
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def _apply(m, d, ext_m):
    """Set column extension and gear the sprocket spin to it (x = r_p·θ)."""
    ext_m = float(np.clip(ext_m, 0.0, STROKE))
    _set(m, d, "deploy", ext_m)
    _set(m, d, "spin", ext_m / RP)
    mujoco.mj_forward(m, d)


def _stroke_of(phase):
    """Deploy 0 -> full -> 0 over one cycle."""
    return STROKE * 0.5 * (1 - np.cos(phase))


def view(cycles_per_s=0.15):
    from mujoco import viewer as mjv
    path = build_xml()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    dt = 1.0 / 60.0
    print(f"\nviewer: zip-chain actuator | deploys 0 -> {P.stroke_mm:.0f} mm "
          f"(push {P.push_capacity_N(P.retract_free_mm):.0f} N retracted -> "
          f"{P.push_capacity_N(P.retract_free_mm+P.stroke_mm):.0f} N at full ext)")
    print("  close window to exit")
    with mjv.launch_passive(m, d) as v:
        ph = 0.0
        while v.is_running():
            ph += 2 * pi * cycles_per_s * dt
            _apply(m, d, _stroke_of(ph))
            v.sync()
            time.sleep(dt)


def render(n_frames=48, height=720, width=960):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    path = build_xml()
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    try:
        r = mujoco.Renderer(m, height=height, width=width)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        r = mujoco.Renderer(m, height=height, width=width)

    frames, strokes = [], []
    for i in range(n_frames):
        ph = 2 * pi * i / (n_frames - 1)
        ext = _stroke_of(ph)
        _apply(m, d, ext)
        strokes.append(ext / m_)
        r.update_scene(d, camera="iso")
        frames.append(r.render().copy())

    OUT.mkdir(exist_ok=True)
    gif = OUT / "sim.gif"
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif, frames, fps=18, loop=0)
        print(f"  wrote {gif}")
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=55, loop=0)
        print(f"  wrote {gif}  (via PIL)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [n_frames // 8, n_frames // 4, 3 * n_frames // 8, n_frames // 2,
             5 * n_frames // 8, 3 * n_frames // 4]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for a, k in zip(ax.ravel(), picks):
        s = strokes[k]
        push = P.push_capacity_N(P.retract_free_mm + s)
        a.imshow(frames[k]); a.set_axis_off()
        a.set_title(f"stroke {s:.0f} mm → push {push:.0f} N", fontsize=9)
    fig.suptitle("Zip-chain linear actuator — column deploys from the head; push ceiling "
                 "falls with extension (buckling)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "sim_montage.png", dpi=110)
    print(f"  wrote {OUT/'sim_montage.png'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "render":
        render()
    else:
        view(cycles_per_s=float(arg) if arg else 0.15)
