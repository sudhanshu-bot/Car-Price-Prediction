from pathlib import Path

import pandas as pd

from model_pipeline import DATA_PATH, load_clean_data


BASE_DIR = Path(__file__).resolve().parent
required = ["cars_ds_final_2021.csv", "cars_report (1).ipynb", "model_pipeline.py", "app.py", "streamlit_app.py", "README.md"]
missing = [name for name in required if not (BASE_DIR / name).exists()]
if missing:
    raise SystemExit(f"Missing project files: {missing}")

raw = pd.read_csv(DATA_PATH)
clean = load_clean_data(DATA_PATH)
assert len(raw) > 0 and len(clean) > 0
assert clean["Ex-Showroom_Price"].notna().all()
print(f"Verification passed: {len(raw):,} raw rows -> {len(clean):,} clean rows")