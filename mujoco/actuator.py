"""
Electromechanical spec for the cycloidal actuator: motor constants + gearbox ratio
-> joint-level torque/speed/inertia numbers that MuJoCo (Layer A) needs.

This is the bridge between the CAD Params (cycloidal/drive.py) and the sim.
NOTE: efficiency here is a LUMPED INPUT, not a derived value — MuJoCo cannot
discover it. The real number comes from the Layer B analytical model + torque-meter.
"""

import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path

# pull the gearbox ratio + efficiency model straight from the CAD side so they never drift
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cycloidal"))
from drive import Params  # noqa: E402
from efficiency import predict_params, drag_out, eta_at_load  # noqa: E402


@dataclass
class MotorSpec:
    """Goolsky / Surpass 2204 1400KV 14-pole outrunner (measured / datasheet)."""
    kv: float = 1400.0          # rpm / V
    max_current: float = 13.0   # A (burst)
    resistance: float = 0.307   # ohm
    voltage: float = 11.1       # 3S nominal; range 7-15 V
    rotor_inertia: float = 1.5e-6   # kg·m²  (estimate: ~12 g bell at ~11 mm eff. radius)

    @property
    def kt(self) -> float:
        """Torque constant, N·m/A = 60 / (2*pi*KV)."""
        return 60.0 / (2.0 * pi * self.kv)

    @property
    def ke(self) -> float:
        """Back-EMF constant, V·s/rad. In SI, Ke == Kt."""
        return self.kt

    @property
    def no_load_speed(self) -> float:
        """Motor no-load speed, rad/s, at `voltage`."""
        return self.kv * self.voltage * 2.0 * pi / 60.0

    @property
    def stall_current(self) -> float:
        """Current at zero speed, full voltage (V/R) — far above the rating."""
        return self.voltage / self.resistance

    @property
    def corner_speed(self) -> float:
        """Motor speed (rad/s) where the back-EMF-natural current falls to max_current.
        Below this the motor is current-limited (flat torque); above, voltage-limited."""
        return (self.voltage - self.max_current * self.resistance) / self.ke

    def current_at(self, omega_motor: float, throttle: float = 1.0) -> float:
        """Back-EMF-limited current at motor speed `omega_motor`, throttle ∈ [-1, 1].
            I = (throttle·V − Ke·ω) / R,  clamped to ±max_current."""
        v = throttle * self.voltage
        I = (v - self.ke * omega_motor) / self.resistance
        return max(-self.max_current, min(self.max_current, I))


@dataclass
class ActuatorSpec:
    ratio: float
    eta_inf: float              # asymptotic (high-load) efficiency from the Layer-B model
    drag_out: float             # N·m no-load drag referred to the output (-> joint frictionloss)
    kt: float
    # joint-side (output) numbers
    peak_torque: float          # N·m net at burst current (drag subtracted)
    cont_torque: float          # N·m rough continuous (thermal ~1/3 burst)
    no_load_speed: float        # rad/s at the output
    reflected_inertia: float    # kg·m²  (armature seen at the joint)
    torque_per_amp: float       # N·m/A at the joint  (== gear for a MuJoCo motor, uses eta_inf)

    def eta_at(self, T_out: float) -> float:
        """Load-dependent efficiency at output torque T_out (the curve the sim realizes)."""
        return eta_at_load(T_out, self.eta_inf, self.drag_out)

    def torque_at_speed(self, omega_out: float, motor: "MotorSpec", throttle: float = 1.0) -> float:
        """Available NET output torque at output speed omega_out, including back-EMF
        current limiting and no-load drag. The realistic torque-speed envelope."""
        I = motor.current_at(omega_out * self.ratio, throttle)
        return max(0.0, self.torque_per_amp * I - self.drag_out)

    @classmethod
    def from_motor(cls, motor: MotorSpec, p: Params,
                   efficiency: float = None, cont_fraction: float = 0.33):
        N = p.ratio
        eta_inf = efficiency if efficiency is not None else predict_params(p)
        d_out = drag_out(p)
        tpa = motor.kt * N * eta_inf                  # asymptotic torque/amp (MuJoCo gear)
        peak = tpa * motor.max_current - d_out        # net output torque (drag subtracted)
        cont = tpa * (motor.max_current * cont_fraction) - d_out
        return cls(
            ratio=N,
            eta_inf=eta_inf,
            drag_out=d_out,
            kt=motor.kt,
            peak_torque=peak,
            cont_torque=cont,
            no_load_speed=motor.no_load_speed / N,
            reflected_inertia=motor.rotor_inertia * N * N,
            torque_per_amp=tpa,
        )

    def report(self, motor: MotorSpec):
        rpm = self.no_load_speed * 60 / (2 * pi)
        print("\n=== Actuator spec (Goolsky 2204 + cycloidal) ===")
        print(f"ratio ............. {self.ratio:.0f}:1   η∞ (high-load) {self.eta_inf:.0%}   "
              f"no-load drag {self.drag_out*1000:.0f} mN·m (output)")
        print(f"motor Kt .......... {self.kt*1000:.2f} mN·m/A   no-load {motor.no_load_speed*60/2/pi:.0f} rpm @ {motor.voltage} V")
        print(f"PEAK torque ....... {self.peak_torque:.3f} N·m  (@ {motor.max_current:.0f} A burst, net of drag)")
        print(f"cont. torque ~..... {self.cont_torque:.3f} N·m  (thermal-limited estimate)")
        print(f"no-load out speed . {self.no_load_speed:.1f} rad/s  ({rpm:.0f} rpm)")
        print(f"reflected inertia . {self.reflected_inertia*1e6:.1f}  g·cm² ×1e3  ({self.reflected_inertia:.2e} kg·m²)")
        print(f"torque / amp ...... {self.torque_per_amp:.4f} N·m/A  (MuJoCo motor gear)")
        # load-dependent efficiency curve (the torque-based model)
        print("efficiency vs load (η rises with torque toward η∞):")
        for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
            T = self.peak_torque * frac
            print(f"    T_out={T:5.3f} N·m ({frac*100:3.0f}% peak) -> η = {self.eta_at(T)*100:4.0f}%")
        # torque-speed envelope (back-EMF): flat-then-droop to zero at no-load speed
        corner_out = motor.corner_speed / self.ratio
        print(f"torque-speed (full throttle): current-limited below {corner_out*60/2/pi:.0f} rpm, "
              f"then droops to 0 at {self.no_load_speed*60/2/pi:.0f} rpm")
        for rpm in (0, 250, 500, 1000, 1250, 1500):
            w = rpm * 2 * pi / 60
            print(f"    {rpm:5d} rpm -> {self.torque_at_speed(w, motor):5.3f} N·m")
        for L in (0.10, 0.15):
            print(f"max static payload @ {L*1000:.0f} mm = {self.peak_torque/(9.81*L)*1000:.0f} g (peak) / "
                  f"{self.cont_torque/(9.81*L)*1000:.0f} g (cont.)")
        print()


if __name__ == "__main__":
    m = MotorSpec()
    spec = ActuatorSpec.from_motor(m, Params())
    spec.report(m)
