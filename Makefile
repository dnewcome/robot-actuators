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
	@echo "  make cycloidal              build the compound planetary+cycloidal drive"

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
