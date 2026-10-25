CONFIG ?= config.yaml
VENV   ?= .venv
PY     := $(VENV)/bin/python
IQP    := $(PY) -m iqp --config $(CONFIG)

STEPS := setup download convert validate_data prepare_data build_imatrix quantize fertility \
         run_kld run_milu run_mt aggregate plot report

.PHONY: all venv test dryrun plan status analysis clean-dryrun $(STEPS)

all: venv
	$(PY) -m iqp all --config $(CONFIG)

venv: $(PY)
$(PY):
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

$(STEPS): venv
	$(PY) -m iqp $@ --config $(CONFIG)

analysis: venv
	$(PY) -m iqp analysis --config $(CONFIG)

plan status: venv
	$(PY) -m iqp $@ --config $(CONFIG)

test: venv
	$(PY) -m pytest -q

dryrun: venv
	$(PY) -m iqp all --config $(CONFIG) --dry-run

clean-dryrun:
	rm -rf work_dryrun results_dryrun logs_dryrun
