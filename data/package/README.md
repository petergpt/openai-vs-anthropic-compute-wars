# OpenAI / Anthropic Open Data Package

This package is the full reconciliation layer behind the published OpenAI / Anthropic compute chart.

If you want the quick visual summary, use the website's `Data` tab.
If you want to audit the numbers directly, use the files in this package.

Keys:
- `point_id`: one plotted monthly point on the chart
- `anchor_id`: one yearly or current anchor used to shape the line
- `data_center_id`: one datacenter in the site-backed floor
- `timeline_row_id`: one raw datacenter timeline row
- `evidence_id`: one public evidence item from the evidence pack

The main concepts are:
- `total_gw`: the plotted total for a company at a given point
- `site_floor_gw`: the named-site layer that can be tied to datacenter rows
- `uplift_gw`: the remaining non-site compute needed to reach the total

In other words:

`total_gw = site_floor_gw + uplift_gw`

How to reconcile one chart point:
1. Start with `derived/chart_points_monthly.csv` and pick a `point_id`.
2. Join to `derived/chart_points_monthly_formula.csv` on `point_id` to see which anchor interval and pacing formula produced the total.
3. Join to `derived/site_floor_components_monthly.csv` on `point_id` to see which datacenters make up the site floor for that month.
4. Join each site component to `derived/data_center_timeline_rows.csv` on `timeline_row_id` to inspect the exact timeline row being used.
5. Join each datacenter to `derived/data_center_registry.csv` and `derived/data_center_selected_sources.csv` on `data_center_id` to inspect the datacenter-level source list.
6. Join the point's `anchor_id` (or its `start_anchor_id` / `end_anchor_id`) to `derived/yearly_anchor_registry.csv`.
7. Join anchors to `derived/anchor_evidence_links.csv`, then to `source_inputs/openai_anthropic_evidence_pack.csv` on `evidence_id`, to inspect the public evidence behind the anchor.

Interpretation:
- `site_floor_gw` is the hard site-backed layer.
- `uplift_gw` is the non-site residual between the plotted total and the site floor.
- Non-anchor monthly points are interpolated between anchor totals using the chart's pacing formula.
- Future anchors from 2026 onward include analyst judgment. They are transparent here, but they are not direct company-reported totals.

Suggested reading order:
1. `derived/yearly_anchor_registry.csv` for the anchor totals.
2. `derived/site_floor_components_monthly.csv` for the named-site decomposition.
3. `derived/chart_points_monthly.csv` for the final plotted monthly series.
4. `derived/chart_points_monthly_formula.csv` for interpolation logic.
