# Hierarchy-Consistent Marine Species Recognition

Deep-learning pipeline for the FathomNet 2025 FGVC challenge (family → genus → species). Use this repo to download the COCO-formatted data, normalize metadata, and train multi-head models.

## Environment setup

Python 3.10+ recommended.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Place the official FathomNet-2025 assets under `external/fathomnet-2025/`:
- `dataset_train.json`
- `dataset_test.json`
- `download.py`
- `requirements.txt`

Then run:
```bash
make download_real    # pulls imagery referenced by the COCO manifests
# first-time prep: downloads + CSVs
make prep_data
# rerun without redownloading (after initial pull)
PREP_ARGS=--skip-download make prep_data
```

Raw images live in `data/kaggle/images/`. Stratified splits are written to `data/splits/` (train/val/test CSVs).

## Training entry point

```bash
make train  # default smoke run with limited batches
make eval   # placeholder for evaluation/reporting scripts
```

## Make targets

- `make setup` – upgrade pip & install requirements.
- `make download_real` – download actual framegrabs with validation.
- `make prep_data` – build normalized CSVs + label encoders.
- `make train` – run the Lightning training CLI.
- `make eval` – placeholder for evaluation scripts.
- `make format` / `make lint` – run Black, Isort, and Ruff over `src/`.

## Repo layout

```
fathomnet-hcls/
  configs/
    resnet50_threeheads.yaml
  data/
  external/
    fathomnet-2025/
  reports/
  scripts/
  src/
    data/
    models/
    train/
    eval/
    utils/
    losses/
```

Add experiments/results under `reports/` when preparing the final paper.
