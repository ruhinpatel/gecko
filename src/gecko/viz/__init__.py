"""Visualization helpers for gecko."""

from __future__ import annotations

from gecko.viz.io import (
	build_shg_df_from_db,
	load_shg_df_from_csv,
	geometry_map_from_df,
	write_beta_viewer_bundle,
)
from gecko.viz.omega import assign_shg_omega_index
from gecko.viz.fields import (
	ErrorSettings,
	compute_error_fields,
	evaluate_field,
	load_lebedev_grid,
	tensor_from_long,
)
from gecko.viz.state import (
	FIELD_CHOICES,
	METRIC_CHOICES,
	auto_clim,
	default_state,
	metric_style,
)

__all__ = [
	"build_shg_df_from_db",
	"load_shg_df_from_csv",
	"geometry_map_from_df",
	"write_beta_viewer_bundle",
	"assign_shg_omega_index",
	"ErrorSettings",
	"compute_error_fields",
	"evaluate_field",
	"load_lebedev_grid",
	"tensor_from_long",
	"FIELD_CHOICES",
	"METRIC_CHOICES",
	"auto_clim",
	"default_state",
	"metric_style",
]
