PYTHON ?= python3
CONFIG ?= configs/resnet50_threeheads.yaml
PREP_ARGS ?=
TRAIN_EPOCHS ?= 2
LIMIT_TRAIN ?= 200
LIMIT_VAL ?= 60

.PHONY: setup download_real prep_data train eval format lint

setup:
	@echo "Create venv (macOS/Linux): python3 -m venv .venv && source .venv/bin/activate"
	@echo "Create venv (Windows): py -3 -m venv .venv && .venv\\Scripts\\activate"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download_real:
	$(PYTHON) scripts/download_real_images.py --train external/fathomnet-2025/dataset_train.json --test external/fathomnet-2025/dataset_test.json --out data/kaggle/images --workers 16

prep_data:
	$(PYTHON) scripts/prepare_from_coco.py --num-workers 16 --min-per-class 50 $(PREP_ARGS)

train:
	$(PYTHON) -m src.train.train --config $(CONFIG) --limit-train $(LIMIT_TRAIN) --limit-val $(LIMIT_VAL) --epochs $(TRAIN_EPOCHS)

eval:
	$(PYTHON) -m src.eval.report --config $(CONFIG)

format:
	black --line-length 100 src
	isort --profile black --line-length 100 src

lint:
	ruff check src
