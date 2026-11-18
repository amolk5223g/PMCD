# Raw Dataset Pointer (PMPDB)

To avoid duplicating ~350 MB of raw sensor CSV files, this project reads directly from your existing local PMPDB dataset directory:

- **Detected Raw Data Location**: `C:\Users\Amol\Downloads\PMCD\PMPDB\raw-data`
- **Configured Path**: `backend/config.py` → `RAW_DATASET_DIR`
- **Contents**: 90 session subdirectories (`P01_1` through `P49_2`) across 49 unique participants, each containing `<ID>.csv`, `<ID>_runUp.csv`, `noPainThreshold.txt`, and `severePainThreshold.txt`.

Processed outputs (`features.parquet`, `features.csv`, `windows_dl.npz`, `feature_metadata.json`) are saved in `../processed/`, and audit metadata (`dataset_summary.json`, `recordings_catalog.csv`, `feature_pain_correlations.csv`) are saved in `../metadata/`.

