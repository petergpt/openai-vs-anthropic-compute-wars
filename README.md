# OpenAI vs Anthropic Compute Wars

This repository contains:
- a public website and interactive chart comparing OpenAI and Anthropic compute buildout
- the source-backed datacenter inputs behind that chart
- a reconciliable open-data package that shows how the published totals are constructed

The goal is simple: let someone open the chart quickly, inspect the underlying evidence, and trace the published numbers back to source-backed inputs without digging through internal working files.

![March 2026 current phase](docs/images/current-phase.png)

## Start Here

- [docs/index.html](/Users/peter/projects/data-center-capacity-allocation-20260327/docs/index.html) is the main static-site entry point.
- [docs/openai-anthropic-training-story.html](/Users/peter/projects/data-center-capacity-allocation-20260327/docs/openai-anthropic-training-story.html) is the direct chart file.
- [data/package/README.md](/Users/peter/projects/data-center-capacity-allocation-20260327/data/package/README.md) explains the full open-data package and how to reconcile one chart point.
- [data/raw/data_centers.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/raw/data_centers.csv) is the main facility table. Each row includes notes and source links.
- [data/raw/data_center_timelines.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/raw/data_center_timelines.csv) is the dated buildout record behind each facility.
- [data/raw/openai_anthropic_major_model_releases.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/raw/openai_anthropic_major_model_releases.csv) is the release table used for the training-window overlays.
- [data/derived/openai_anthropic_publishable_view.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/derived/openai_anthropic_publishable_view.csv) is the clean yearly anchor view used for the published chart.
- [data/derived/company_capacity_by_snapshot.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/derived/company_capacity_by_snapshot.csv) is the site-backed rollup by company and snapshot date.
- [data/package/](/Users/peter/projects/data-center-capacity-allocation-20260327/data/package) is the self-contained package with source inputs, derived tables, and a manifest.

## What Is Live

- The website at [compute-wars.surge.sh](https://compute-wars.surge.sh) publishes the chart, the interactive Data tab, and the downloadable chart breakdown CSV.
- GitHub contains the full source tree, including the complete package under [data/package/](/Users/peter/projects/data-center-capacity-allocation-20260327/data/package).
- The website does not currently expose the raw package tables directly as browsable files; those live in the repository.

## How The Numbers Work

At a high level, each published total is treated as:

`total compute = named sites + other compute`

- `named sites` is the hard site-backed layer that can be tied to datacenter records and timeline rows.
- `other compute` is the non-site residual needed to match the published estimate for that point.
- Monthly values are then interpolated between anchor points using the pacing formulas in the package.

The website's `Data` tab is the light, human-readable view. The package in [data/package/](/Users/peter/projects/data-center-capacity-allocation-20260327/data/package) is the audit trail.

## Rebuild

`cd /Users/peter/projects/data-center-capacity-allocation-20260327 && python3 scripts/build_release.py`

## Data And Sources

The easiest raw entry point is [data/raw/data_centers.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/raw/data_centers.csv).

Every named site keeps its supporting links in the `Selected Sources` column, and the dated buildout record lives in [data/raw/data_center_timelines.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/raw/data_center_timelines.csv).

For a packaged, public-facing reconciliation path, start from [data/package/derived/chart_points_monthly.csv](/Users/peter/projects/data-center-capacity-allocation-20260327/data/package/derived/chart_points_monthly.csv) and follow the joins described in [data/package/README.md](/Users/peter/projects/data-center-capacity-allocation-20260327/data/package/README.md).

## License

- Code in this repository is released under the ISC license. See `LICENSE`.
- The underlying Epoch AI data is available under the Creative Commons Attribution 4.0 license. See `DATA-LICENSE.md`.

## Citation

Epoch AI, "Frontier Data Centers". Published online at `https://epoch.ai/data/data-centers`.
