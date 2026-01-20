from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from gecko.viz.io import build_shg_df_from_db, load_shg_df_from_csv
from migration.viz import application as legacy

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--db-dir", dest="db_dir", default=None, help="Root directory of calculations")
	parser.add_argument("--csv", dest="csv", default=None, help="Path to shg_ijk.csv")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=9010)
	parser.add_argument("--mol-file", dest="mol_file", default=None)
	parser.add_argument("--mol-dir", dest="mol_dir", default=None)
	parser.add_argument("--mol-map", dest="mol_map", default=None)
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = _parse_args(argv)

	if not args.csv and not args.db_dir:
		raise SystemExit("Provide --csv or --db-dir")

	if args.csv:
		csv_path = Path(args.csv).expanduser().resolve()
	else:
		df = build_shg_df_from_db(Path(args.db_dir))
		tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
		csv_path = Path(tmp.name)
		df.to_csv(csv_path, index=False)

	argv_out = ["--shg-csv", str(csv_path), "--host", str(args.host), "--port", str(args.port)]
	if args.mol_file:
		argv_out.extend(["--mol-file", str(args.mol_file)])
	if args.mol_dir:
		argv_out.extend(["--mol-dir", str(args.mol_dir)])
	if args.mol_map:
		argv_out.extend(["--mol-map", str(args.mol_map)])

	return legacy.main(argv_out)
