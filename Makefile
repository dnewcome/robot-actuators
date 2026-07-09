# Robot actuators — build (CAD -> STEP/STL) and sim (kinematic MuJoCo viewer) targets.
# All parts use the in-repo virtualenv (build123d + mujoco live there).
PY := .venv/bin/python

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "targets:"
	@echo "  make cycloidal-center       build the straddle-carrier centre-output cycloidal"
	@echo "  make sim-cycloidal-center   build + open its kinematic MuJoCo viewer"
	@echo "  make planetary              build the standalone planetary reducer"
	@echo "  make sim-planetary          build + open its kinematic MuJoCo viewer"
	@echo "  make planetary-inverted     build the inverted carrier-out planetary (NEMA-17 in/out)"
	@echo "  make sim-planetary-inverted build + open its kinematic MuJoCo viewer"
	@echo "  make cycloidal-2stage       build the two-stage compound cycloidal (body output, yoke)"
	@echo "  make sim-cycloidal-2stage   build + open its kinematic MuJoCo viewer (121:1)"
	@echo "  make cycloidal              build the compound planetary+cycloidal drive"
	@echo "  make motor                  EM FEA: mesh + flux-density field of the torque motor"
	@echo "  make motor-curves           EM FEA: + cogging & Kt sweeps (slower)"
	@echo "  make linear                 piezo-hydraulic linear actuator sizing + F-v envelope"
	@echo "  make linear-eo              add the electroosmotic route + piezo-vs-EO comparison"
	@echo "  make rail-encoder           capacitive vernier scale (digital-caliper) resolution model"
	@echo "  make rail-servo             control proof: TT gearmotor + cap scale cancels backlash"
	@echo "  make rail-cad               build the linear rail servo CAD (STEP/STL parts)"
	@echo "  make sim-rail               build CAD + open its kinematic MuJoCo viewer"
	@echo "  make drawwire               draw-wire (tape-measure) encoder resolution model"
	@echo "  make corexy                 CoreXY proof: draw-wire loop cancels belt backlash/stretch"
	@echo "  make pcb-motor              axial-flux PCB-stator motor model + reduction-need calculator"
	@echo "  make pcb-motor-fea          2D unrolled magnetostatic FEA cross-check of the air-gap flux"
	@echo "  make pcb-motor-benchmark    calibrate the motor model vs a measured PCB motor (Wang 2025)"

# --- centre-output cycloidal ------------------------------------------------
.PHONY: cycloidal-center
cycloidal-center:
	$(PY) cycloidal-center/drive.py

.PHONY: sim-cycloidal-center
sim-cycloidal-center: cycloidal-center
	$(PY) cycloidal-center/sim.py $(REV)

# --- standalone planetary reducer -------------------------------------------
.PHONY: planetary
planetary:
	$(PY) planetary/reducer.py

.PHONY: sim-planetary
sim-planetary: planetary
	$(PY) planetary/sim.py $(REV)

# --- inverted (carrier-out) planetary, NEMA-17 in/out -----------------------
.PHONY: planetary-inverted
planetary-inverted:
	$(PY) planetary-inverted/drive.py

.PHONY: sim-planetary-inverted
sim-planetary-inverted: planetary-inverted
	$(PY) planetary-inverted/sim.py $(REV)

# --- compound planetary+cycloidal drive -------------------------------------
.PHONY: cycloidal
cycloidal:
	$(PY) cycloidal/drive.py

# --- two-stage compound cycloidal (body output, external-yoke straddle) ------
.PHONY: cycloidal-2stage
cycloidal-2stage:
	$(PY) cycloidal-2stage/drive.py

.PHONY: sim-cycloidal-2stage
sim-cycloidal-2stage: cycloidal-2stage
	$(PY) cycloidal-2stage/sim.py $(REV)

# --- custom torque-motor electromagnetic FEA (gmsh + scikit-fem) ------------
.PHONY: motor
motor:
	$(PY) motor/fea.py

.PHONY: motor-curves
motor-curves:
	$(PY) motor/fea.py curves

# --- piezo-hydraulic linear actuator (first-order sizing model) -------------
.PHONY: linear
linear:
	$(PY) linear/actuator.py

.PHONY: linear-eo
linear-eo:
	$(PY) linear/actuator.py eo

# --- linear rail servo: crappy TT gearmotor + capacitive vernier scale -------
.PHONY: rail-encoder
rail-encoder:
	$(PY) linear-rail-servo/encoder.py

.PHONY: rail-servo
rail-servo:
	$(PY) linear-rail-servo/servo.py

.PHONY: rail-cad
rail-cad:
	$(PY) linear-rail-servo/rail.py

.PHONY: sim-rail
sim-rail: rail-cad
	$(PY) linear-rail-servo/sim.py $(REV)

# --- CoreXY stage on TT motors, closed on draw-wire (string-pot) encoders -----
.PHONY: drawwire
drawwire:
	$(PY) corexy/drawwire.py

.PHONY: corexy
corexy:
	$(PY) corexy/corexy.py

# --- axial-flux PCB-stator motor (analytical model + reduction calculator) ----
.PHONY: pcb-motor
pcb-motor:
	$(PY) pcb-motor/motor.py

.PHONY: pcb-motor-fea
pcb-motor-fea:
	$(PY) pcb-motor/fea.py

.PHONY: pcb-motor-benchmark
pcb-motor-benchmark:
	$(PY) pcb-motor/benchmark.py
