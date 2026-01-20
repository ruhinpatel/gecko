from gecko.core.iterators import iter_calc_dirs
from gecko.recipes.shg_csv import build_beta_table
from pathlib import Path

base_dir = Path("tests/fixtures/calc_nlo_beta/NLO")
data_dir = base_dir
mol_dir = base_dir / "molecules"
dirs = list(iter_calc_dirs(data_dir))
df = build_beta_table(dirs,mol_dir=mol_dir, require_geometry=True, verbose=True)
print(df.head())
print("rows:", len(df))

# Save to CSV
df.to_csv("beta_table.csv", index=False)
