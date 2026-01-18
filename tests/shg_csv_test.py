from gecko.core.iterators import iter_calc_dirs
from gecko.recipes.shg_csv import build_beta_table

dirs = list(iter_calc_dirs("migration/templates/outputs"))
df = build_beta_table(dirs)
print(df.head())
print("rows:", len(df))

# Save to CSV
df.to_csv("beta_table.csv", index=False)
