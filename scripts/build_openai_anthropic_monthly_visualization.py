from __future__ import annotations

import calendar
import csv
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import analyze_allocations as allocations
import paths

PUBLISHABLE_VIEW = paths.OPENAI_ANTHROPIC_PUBLISHABLE_VIEW_CSV
MONTHLY_CSV = paths.OPENAI_ANTHROPIC_MONTHLY_SERIES_CSV
MONTHLY_HTML = paths.MONTHLY_VISUALIZATION_HTML
MODEL_RELEASES_CSV = paths.OPENAI_ANTHROPIC_MAJOR_MODEL_RELEASES_SOURCE_CSV
MODEL_OVERLAY_CSV = paths.OPENAI_ANTHROPIC_MODEL_OVERLAY_EVENTS_CSV

COMPANIES = ("OpenAI", "Anthropic")
START_MONTH = date(2023, 1, 31)
END_MONTH = date(2029, 12, 31)
MODEL_TRAINING_LAG_MONTHS = 6

DATE_BY_ROW_KEY = {
    "2023": date(2023, 12, 31),
    "2024": date(2024, 12, 31),
    "2025": date(2025, 12, 31),
    "2026_current": date(2026, 3, 31),
    "2026_year_end": date(2026, 12, 31),
    "2027_year_end": date(2027, 12, 31),
    "2028_year_end_floor": date(2028, 12, 31),
    "2029_year_end_floor": date(2029, 12, 31),
}

ANCHOR_LABELS = {
    "2023": "2023 year-end",
    "2024": "2024 year-end",
    "2025": "2025 year-end",
    "2026_current": "March 2026 current",
    "2026_year_end": "2026 year-end",
    "2027_year_end": "2027 year-end",
    "2028_year_end_floor": "2028 year-end",
    "2029_year_end_floor": "2029 year-end",
}

COMPANY_COLORS = {
    "OpenAI": "#101010",
    "Anthropic": "#db5c2b",
}


@dataclass(frozen=True)
class Anchor:
    company: str
    date_value: date
    value_gw: float
    label: str
    estimate_type: str
    note: str


@dataclass(frozen=True)
class ModelOverlay:
    company: str
    model_key: str
    model_name: str
    short_label: str
    display_label: str
    release_date: date
    estimated_training_start_date: date
    source_url: str
    source_title: str


def month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def shift_months(value: date, months: int) -> date:
    total_months = value.year * 12 + (value.month - 1) + months
    target_year = total_months // 12
    target_month = total_months % 12 + 1
    last_day = calendar.monthrange(target_year, target_month)[1]
    return date(target_year, target_month, min(value.day, last_day))


def month_ends(start: date, end: date) -> list[date]:
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 31)
        else:
            current = month_end(current.year, current.month + 1)
    return out


def smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def fmt_gw(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def overlay_display_label(model_key: str, short_label: str) -> str:
    custom = {
        "openai:gpt-4": "GPT-4",
        "openai:gpt-4-turbo": "GPT-4T",
        "openai:gpt-4o": "GPT-4o",
        "openai:o1-preview": "o1p",
        "openai:o1": "o1",
        "openai:gpt-4.5": "GPT-4.5",
        "openai:gpt-4.1": "GPT-4.1",
        "openai:o3": "o3",
        "openai:gpt-5": "GPT-5",
        "openai:gpt-5.1": "GPT-5.1",
        "openai:gpt-5.2": "GPT-5.2",
        "openai:gpt-5.4": "GPT-5.4",
        "anthropic:claude-2": "C2",
        "anthropic:claude-2.1": "2.1",
        "anthropic:claude-3-opus": "3O",
        "anthropic:claude-3.5-sonnet": "3.5S",
        "anthropic:claude-3.5-sonnet-new": "3.5S+",
        "anthropic:claude-3.7-sonnet": "3.7S",
        "anthropic:claude-opus-4": "O4",
        "anthropic:claude-opus-4.1": "O4.1",
        "anthropic:claude-sonnet-4.5": "S4.5",
        "anthropic:claude-opus-4.5": "O4.5",
        "anthropic:claude-opus-4.6": "O4.6",
    }
    return custom.get(model_key, short_label)


def read_publishable_anchors() -> dict[str, list[Anchor]]:
    anchors: dict[str, list[Anchor]] = {company: [] for company in COMPANIES}
    with PUBLISHABLE_VIEW.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_key = row["year"]
            anchor_date = DATE_BY_ROW_KEY[row_key]
            note = row["summary_note"].strip()
            for company in COMPANIES:
                key = company.lower()
                anchors[company].append(
                    Anchor(
                        company=company,
                        date_value=anchor_date,
                        value_gw=float(row[f"{key}_point_estimate"]),
                        label=ANCHOR_LABELS[row_key],
                        estimate_type=row[f"{key}_type"],
                        note=note,
                    )
                )
    return anchors


def read_model_overlays() -> list[ModelOverlay]:
    company_by_org = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }
    overlays: list[ModelOverlay] = []
    with MODEL_RELEASES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            company = company_by_org[str(row["org"]).strip().lower()]
            release_date = date.fromisoformat(str(row["release_date"]))
            overlays.append(
                ModelOverlay(
                    company=company,
                    model_key=str(row["model_key"]),
                    model_name=str(row["model_name"]),
                    short_label=str(row["short_label"]),
                    display_label=overlay_display_label(
                        str(row["model_key"]),
                        str(row["short_label"]),
                    ),
                    release_date=release_date,
                    estimated_training_start_date=shift_months(
                        release_date,
                        -MODEL_TRAINING_LAG_MONTHS,
                    ),
                    source_url=str(row["source_url"]),
                    source_title=str(row["source_title"]),
                )
            )
    overlays.sort(key=lambda item: (item.release_date, item.company, item.model_name))
    return overlays


def load_mappings() -> tuple[dict[str, dict], dict[str, list[allocations.TimelineRow]]]:
    centers = allocations.read_csv(paths.DATA_CENTERS_CSV)
    rows_by_center = allocations.load_timeline_rows(paths.DATA_CENTER_TIMELINES_CSV)
    mapping_rows: list[dict] = []
    for row in centers:
        mapping, _ = allocations.build_center_mapping(row)
        mapping_rows.append(mapping)
    return {row["data_center"]: row for row in mapping_rows}, rows_by_center


def summarize_center_delta(delta_gw: float, center: str) -> str:
    sign = "+" if delta_gw >= 0 else ""
    return f"{center} {sign}{delta_gw:.3f} GW"


def build_monthly_floor(
    mapping_by_center: dict[str, dict],
    rows_by_center: dict[str, list[allocations.TimelineRow]],
    dates: Iterable[date],
) -> tuple[dict[str, dict[date, float]], dict[str, dict[date, list[str]]]]:
    floor_by_company: dict[str, dict[date, float]] = {company: {} for company in COMPANIES}
    events_by_company: dict[str, dict[date, list[str]]] = {company: {} for company in COMPANIES}
    previous_center_values: dict[str, dict[str, float]] = {
        company: {} for company in COMPANIES
    }

    for target_date in dates:
        totals = {company: 0.0 for company in COMPANIES}
        current_center_values = {company: {} for company in COMPANIES}
        current_events = {company: [] for company in COMPANIES}

        for center in sorted(rows_by_center):
            allocation_rows = allocations.allocate_center_snapshot(
                mapping_by_center[center],
                rows_by_center[center],
                target_date,
                [],
            )
            for allocation_row in allocation_rows:
                company = allocation_row["allocated_user"]
                if company not in COMPANIES:
                    continue
                value_gw = allocation_row["power_mw"] / 1000.0
                current_center_values[company][center] = value_gw
                totals[company] += value_gw

        for company in COMPANIES:
            for center, current_value in current_center_values[company].items():
                previous_value = previous_center_values[company].get(center, 0.0)
                delta = current_value - previous_value
                if abs(delta) > 1e-9:
                    current_events[company].append(summarize_center_delta(delta, center))

            for center, previous_value in previous_center_values[company].items():
                if center not in current_center_values[company] and abs(previous_value) > 1e-9:
                    current_events[company].append(
                        summarize_center_delta(-previous_value, center)
                    )

            previous_center_values[company] = current_center_values[company]
            floor_by_company[company][target_date] = totals[company]
            events_by_company[company][target_date] = current_events[company]

    return floor_by_company, events_by_company


def interval_weight(total_delta: float, floor_delta: float, anchor_date: date) -> float:
    if total_delta <= 1e-9:
        return 0.18
    ratio = max(0.0, min(1.0, floor_delta / total_delta))
    weight = 0.18 + 0.54 * ratio
    if anchor_date.year <= 2024:
        weight *= 0.72
    elif anchor_date.year == 2025:
        weight *= 0.88
    return max(0.14, min(0.72, weight))


def pace_company_totals(
    company: str,
    months: list[date],
    floor_by_month: dict[date, float],
    anchors: list[Anchor],
) -> list[dict[str, object]]:
    real_anchor_dates = {anchor.date_value for anchor in anchors}
    anchor_map = {anchor.date_value: anchor for anchor in anchors}
    baseline = Anchor(
        company=company,
        date_value=months[0],
        value_gw=0.0,
        label="Synthetic January 2023 baseline",
        estimate_type="synthetic_baseline",
        note="The monthly line starts from a synthetic zero baseline in January 2023 so the first anchored year-end point can be paced inside the year.",
    )
    all_anchors = [baseline, *anchors]
    rows: list[dict[str, object]] = []

    for start_anchor, end_anchor in zip(all_anchors, all_anchors[1:]):
        interval_months = [
            target_date
            for target_date in months
            if start_anchor.date_value <= target_date <= end_anchor.date_value
        ]
        if len(interval_months) < 2:
            continue

        start_total = start_anchor.value_gw
        end_total = end_anchor.value_gw
        total_delta = end_total - start_total
        start_floor = floor_by_month[start_anchor.date_value]
        end_floor = floor_by_month[end_anchor.date_value]
        floor_delta = max(0.0, end_floor - start_floor)
        blend_weight = interval_weight(total_delta, floor_delta, end_anchor.date_value)

        for idx, target_date in enumerate(interval_months):
            if target_date == start_anchor.date_value and rows:
                continue

            floor_gw = floor_by_month[target_date]

            if target_date == start_anchor.date_value:
                total_gw = max(start_total, floor_gw)
                estimate_basis = "synthetic_baseline"
                anchor_label = ""
                anchor_note = start_anchor.note
                anchor_target_gw = ""
                pacing_interval = f"{start_anchor.label} -> {end_anchor.label}"
            else:
                progress = idx / (len(interval_months) - 1)
                smooth_progress = smoothstep(progress)
                if floor_delta > 1e-9:
                    floor_progress = max(
                        0.0, min(1.0, (floor_gw - start_floor) / floor_delta)
                    )
                else:
                    floor_progress = smooth_progress
                composite_progress = (1.0 - blend_weight) * smooth_progress + blend_weight * floor_progress
                total_gw = start_total + total_delta * composite_progress
                total_gw = max(total_gw, floor_gw)
                if target_date in real_anchor_dates:
                    estimate_basis = "anchored_estimate"
                    anchor_label = anchor_map[target_date].label
                    anchor_note = anchor_map[target_date].note
                    anchor_target_gw = fmt_gw(anchor_map[target_date].value_gw)
                else:
                    estimate_basis = "paced_month"
                    anchor_label = ""
                    anchor_note = ""
                    anchor_target_gw = ""
                pacing_interval = f"{start_anchor.label} -> {end_anchor.label}"

            rows.append(
                {
                    "month_end": target_date.isoformat(),
                    "company": company,
                    "total_gw": round(total_gw, 6),
                    "floor_gw": round(floor_gw, 6),
                    "uplift_gw": round(max(0.0, total_gw - floor_gw), 6),
                    "estimate_basis": estimate_basis,
                    "anchor_label": anchor_label,
                    "anchor_target_gw": anchor_target_gw,
                    "anchor_note": anchor_note,
                    "pacing_interval": pacing_interval,
                    "pacing_blend_weight": round(blend_weight, 4),
                }
            )

    return rows


def build_monthly_rows() -> list[dict[str, object]]:
    months = month_ends(START_MONTH, END_MONTH)
    mapping_by_center, rows_by_center = load_mappings()
    anchors = read_publishable_anchors()
    floor_by_company, events_by_company = build_monthly_floor(mapping_by_center, rows_by_center, months)

    monthly_rows: list[dict[str, object]] = []
    for company in COMPANIES:
        company_rows = pace_company_totals(
            company=company,
            months=months,
            floor_by_month=floor_by_company[company],
            anchors=anchors[company],
        )
        previous_floor = None
        for row in company_rows:
            month_date = date.fromisoformat(str(row["month_end"]))
            floor_events = events_by_company[company][month_date]
            floor_changed = previous_floor is None or abs(row["floor_gw"] - previous_floor) > 1e-9
            row["floor_basis"] = (
                "observed_site_change_month" if floor_changed else "carried_floor"
            )
            row["floor_event_count"] = len(floor_events)
            row["floor_events"] = " | ".join(floor_events)
            previous_floor = float(row["floor_gw"])
            monthly_rows.append(row)

    monthly_rows.sort(key=lambda row: (row["month_end"], row["company"]))
    return monthly_rows


def write_monthly_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "month_end",
        "company",
        "total_gw",
        "floor_gw",
        "uplift_gw",
        "estimate_basis",
        "floor_basis",
        "floor_event_count",
        "floor_events",
        "anchor_label",
        "anchor_target_gw",
        "anchor_note",
        "pacing_interval",
        "pacing_blend_weight",
    ]
    with MONTHLY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def write_model_overlay_csv(overlays: list[ModelOverlay]) -> None:
    fieldnames = [
        "company",
        "model_key",
        "model_name",
        "short_label",
        "display_label",
        "release_date",
        "estimated_training_start_date",
        "training_lag_months",
        "source_url",
        "source_title",
    ]
    with MODEL_OVERLAY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for overlay in overlays:
            writer.writerow(
                {
                    "company": overlay.company,
                    "model_key": overlay.model_key,
                    "model_name": overlay.model_name,
                    "short_label": overlay.short_label,
                    "display_label": overlay.display_label,
                    "release_date": overlay.release_date.isoformat(),
                    "estimated_training_start_date": overlay.estimated_training_start_date.isoformat(),
                    "training_lag_months": MODEL_TRAINING_LAG_MONTHS,
                    "source_url": overlay.source_url,
                    "source_title": overlay.source_title,
                }
            )


def html_template(data: dict[str, object]) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>OpenAI / Anthropic Monthly Compute Buildout</title>
    <style>
      :root {{
        --page: #fbfbfa;
        --surface: #ffffff;
        --surface-strong: #ffffff;
        --ink: #121212;
        --muted: #6a6a66;
        --line: rgba(18, 18, 18, 0.12);
        --openai: {COMPANY_COLORS["OpenAI"]};
        --anthropic: {COMPANY_COLORS["Anthropic"]};
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        min-height: 100svh;
        min-height: 100dvh;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
        background: var(--page);
      }}

      body::before {{
        display: none;
      }}

      .wrap {{
        width: 100%;
        min-height: 100vh;
        min-height: 100svh;
        min-height: 100dvh;
        margin: 0;
        padding: 0;
      }}

      .chart-card {{
        position: relative;
        min-height: 100vh;
        min-height: 100svh;
        min-height: 100dvh;
        height: 100vh;
        height: 100svh;
        height: 100dvh;
        overflow: hidden;
        display: grid;
        grid-template-rows: auto 1fr auto;
        border: none;
        background: var(--surface);
      }}

      .chart-card::before {{
        display: none;
      }}

      .header {{
        position: relative;
        z-index: 1;
        padding: 0.9rem 1.1rem 0.05rem;
      }}

      .title-block h1 {{
        margin: 0;
        font-family: inherit;
        font-size: clamp(1.7rem, 3.2vw, 2.55rem);
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.04em;
      }}

      .title-block p {{
        margin: 0.16rem 0 0;
        max-width: 22rem;
        font-size: 0.9rem;
        color: var(--muted);
        line-height: 1.25;
      }}

      .legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem 0.95rem;
        margin-top: 0.5rem;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        color: var(--muted);
      }}

      .legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
      }}

      .swatch {{
        width: 15px;
        height: 3px;
        border-radius: 0;
        border: none;
      }}

      .swatch.openai {{
        background: var(--openai);
      }}

      .swatch.anthropic {{
        background: var(--anthropic);
      }}

      .controls {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.48rem;
      }}

      .controls-label,
      .lag-note {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        color: var(--muted);
      }}

      .toggle-group {{
        display: inline-flex;
        align-items: center;
        gap: 0;
        padding: 0;
        border: 1px solid rgba(18, 18, 18, 0.12);
        border-radius: 3px;
        background: #ffffff;
        overflow: hidden;
      }}

      .toggle-button {{
        border: none;
        margin: 0;
        padding: 0.34rem 0.64rem;
        border-radius: 0;
        background: transparent;
        color: var(--muted);
        font: inherit;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        cursor: pointer;
        box-shadow: inset -1px 0 0 rgba(18, 18, 18, 0.08);
      }}

      .toggle-button:hover {{
        background: rgba(18, 18, 18, 0.045);
        color: var(--ink);
      }}

      .toggle-button[data-active="true"] {{
        background: #161616;
        color: #ffffff;
        box-shadow: none;
      }}

      .chart-stage {{
        position: relative;
        display: flex;
        align-items: stretch;
        min-height: 0;
        padding: 0 0.9rem;
      }}

      svg {{
        position: relative;
        z-index: 1;
        display: block;
        width: 100%;
        height: 100%;
        min-height: 0;
      }}

      .axis-label,
      .tick-label,
      .year-label {{
        fill: var(--muted);
        font-size: 12px;
        letter-spacing: 0.03em;
      }}

      .axis-label {{
        font-weight: 600;
      }}

      .year-label {{
        font-size: 12px;
        letter-spacing: 0.1em;
        font-weight: 600;
      }}

      .grid-line {{
        stroke: rgba(18, 18, 18, 0.1);
        stroke-width: 1;
        stroke-dasharray: 2 8;
      }}

      .axis-line {{
        stroke: rgba(18, 18, 18, 0.18);
        stroke-width: 1.15;
      }}

      .quarter-line {{
        stroke: rgba(18, 18, 18, 0.08);
        stroke-width: 1;
      }}

      .total-line {{
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 5;
      }}

      .anchor-ring {{
        fill: #ffffff;
        stroke-width: 2.25;
      }}

      .anchor-fill {{
        stroke: none;
      }}

      .right-label {{
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.02em;
      }}

      .model-stem {{
        stroke-width: 1.2;
        opacity: 0.28;
      }}

      .model-marker-release {{
        stroke: #ffffff;
        stroke-width: 1.5;
      }}

      .model-marker-train {{
        fill: #ffffff;
        stroke-width: 1.9;
      }}

      .model-chip {{
        fill: #ffffff;
        stroke: rgba(18, 18, 18, 0.15);
        stroke-width: 1;
      }}

      .model-chip-text {{
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.01em;
      }}

      .footer-line {{
        padding: 0.05rem 1.1rem 0.8rem;
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0;
      }}

      @media (max-width: 920px) {{
        .wrap {{
          min-height: 100vh;
          min-height: 100svh;
          min-height: 100dvh;
        }}

        .header {{
          padding: 0.8rem 0.9rem 0.05rem;
        }}

        .chart-stage {{
          padding: 0 0.45rem;
        }}

        .footer-line {{
          padding: 0.05rem 0.9rem 0.8rem;
        }}

        .controls {{
          gap: 0.55rem;
        }}

        .toggle-button {{
          padding: 0.32rem 0.54rem;
          font-size: 0.72rem;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="wrap">
      <section class="chart-card">
        <div class="header">
          <div class="title-block">
            <h1>OpenAI vs Anthropic compute</h1>
            <p>GW by month. Dots = year-end anchors.</p>
            <div class="legend">
              <span class="legend-item"><span class="swatch openai"></span>OpenAI</span>
              <span class="legend-item"><span class="swatch anthropic"></span>Anthropic</span>
            </div>
            <div class="controls">
              <span class="controls-label">Range</span>
              <div class="toggle-group" role="group" aria-label="Chart range mode">
                <button class="toggle-button" data-range-mode="to-date" data-active="false" type="button">To date</button>
                <button class="toggle-button" data-range-mode="full" data-active="true" type="button">Full</button>
              </div>
              <span class="controls-label">Models</span>
              <div class="toggle-group" role="group" aria-label="Model overlay mode">
                <button class="toggle-button" data-overlay-mode="off" data-active="false" type="button">Off</button>
                <button class="toggle-button" data-overlay-mode="release" data-active="true" type="button">Release</button>
                <button class="toggle-button" data-overlay-mode="train" data-active="false" type="button">Train</button>
              </div>
              <span class="lag-note">Train = release - {MODEL_TRAINING_LAG_MONTHS}m</span>
            </div>
          </div>
        </div>

        <div class="chart-stage">
          <svg id="chart" viewBox="0 0 1320 760" aria-label="OpenAI and Anthropic compute buildout chart"></svg>
        </div>

        <div class="footer-line">
          Source: public estimates paced monthly between year-end anchors; flagship models only.
        </div>
      </section>
    </main>

    <script>
      const DATA = {payload};

      const svg = document.getElementById("chart");
      const stage = document.querySelector(".chart-stage");
      const overlayButtons = Array.from(document.querySelectorAll("[data-overlay-mode]"));
      const rangeButtons = Array.from(document.querySelectorAll("[data-range-mode]"));
      let overlayMode = DATA.default_overlay_mode || "release";
      let rangeMode = DATA.default_range_mode || "full";

      const makeSvg = (tag, attrs = {{}}) => {{
        const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [key, value] of Object.entries(attrs)) {{
          element.setAttribute(key, value);
        }}
        return element;
      }};

      const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
      const isoToTime = (value) => Date.parse(`${{value}}T00:00:00Z`);
      const measureChipWidth = (label, isCompact) => 18 + label.length * (isCompact ? 5.9 : 6.5);

      const months = DATA.months;
      const monthTimes = months.map((month) => isoToTime(month));
      const minTime = monthTimes[0];
      const maxTime = monthTimes[monthTimes.length - 1];
      const currentCutoffTime = isoToTime(DATA.current_cutoff_date);

      const companies = DATA.companies.map((company) => ({{
        ...company,
        points: company.points.map((point) => ({{
          ...point,
          time: isoToTime(point.month_end),
        }})),
      }}));

      const modelsByCompany = new Map(companies.map((company) => [company.name, []]));
      DATA.models.forEach((entry) => {{
        const bucket = modelsByCompany.get(entry.name);
        if (!bucket) {{
          return;
        }}
        entry.models.forEach((model) => {{
          bucket.push({{
            ...model,
            release_time: isoToTime(model.release_date),
            training_start_time: isoToTime(model.estimated_training_start_date),
          }});
        }});
      }});

      const yearPositions = months
        .map((month) => (month.endsWith("-12-31") ? {{ month, time: isoToTime(month) }} : null))
        .filter(Boolean);

      const quarterPositions = months
        .map((month) => {{
          const parsed = new Date(`${{month}}T00:00:00Z`);
          const monthNumber = parsed.getUTCMonth() + 1;
          return [1, 4, 7, 10].includes(monthNumber)
            ? {{ month, time: isoToTime(month) }}
            : null;
        }})
        .filter(Boolean);

      const setOverlayMode = (mode) => {{
        overlayMode = mode;
        overlayButtons.forEach((button) => {{
          button.dataset.active = button.dataset.overlayMode === overlayMode ? "true" : "false";
        }});
        queueRender();
      }};

      overlayButtons.forEach((button) => {{
        button.addEventListener("click", () => setOverlayMode(button.dataset.overlayMode));
      }});

      const setRangeMode = (mode) => {{
        rangeMode = mode;
        rangeButtons.forEach((button) => {{
          button.dataset.active = button.dataset.rangeMode === rangeMode ? "true" : "false";
        }});
        queueRender();
      }};

      rangeButtons.forEach((button) => {{
        button.addEventListener("click", () => setRangeMode(button.dataset.rangeMode));
      }});

      const interpolateValueAtTime = (points, targetTime) => {{
        if (targetTime <= points[0].time) {{
          return points[0].total_gw;
        }}
        if (targetTime >= points[points.length - 1].time) {{
          return points[points.length - 1].total_gw;
        }}
        for (let index = 1; index < points.length; index += 1) {{
          const current = points[index];
          if (targetTime <= current.time) {{
            const previous = points[index - 1];
            const span = current.time - previous.time;
            const progress = span === 0 ? 0 : (targetTime - previous.time) / span;
            return previous.total_gw + (current.total_gw - previous.total_gw) * progress;
          }}
        }}
        return points[points.length - 1].total_gw;
      }};

      function renderModelOverlay(rootGroup, geometry) {{
        if (overlayMode === "off") {{
          return;
        }}

        const appendReleaseMarker = (group, x, y, color) => {{
          group.appendChild(
            makeSvg("circle", {{
              cx: x,
              cy: y,
              r: geometry.isCompact ? 3.7 : 4.1,
              class: "model-marker-release",
              fill: color,
            }})
          );
        }};

        const appendTrainingMarker = (group, x, y, color) => {{
          const size = geometry.isCompact ? 6.4 : 7.2;
          group.appendChild(
            makeSvg("rect", {{
              x: x - size / 2,
              y: y - size / 2,
              width: size,
              height: size,
              rx: 1.2,
              class: "model-marker-train",
              stroke: color,
              transform: `rotate(45 ${{x}} ${{y}})`,
            }})
          );
        }};

        if (geometry.isCompact || geometry.modelBandHeight <= 0) {{
          companies.forEach((company) => {{
            const models = (modelsByCompany.get(company.name) || []).filter((model) => {{
              const targetTime =
                overlayMode === "train" ? model.training_start_time : model.release_time;
              return targetTime >= minTime && targetTime <= geometry.maxVisibleTime;
            }});
            models.forEach((model) => {{
              const releaseX = clamp(geometry.xForTime(model.release_time), geometry.plotLeft, geometry.plotRight);
              const trainX = clamp(geometry.xForTime(model.training_start_time), geometry.plotLeft, geometry.plotRight);
              const releaseY = geometry.yForValue(interpolateValueAtTime(company.points, model.release_time));
              const trainY = geometry.yForValue(interpolateValueAtTime(company.points, model.training_start_time));
              const group = makeSvg("g");
              const title = makeSvg("title");
              title.textContent = `${{model.model_name}} | release ${{model.release_date}} | est. train start ${{model.estimated_training_start_date}}`;
              group.appendChild(title);

              if (overlayMode === "train" || overlayMode === "both") {{
                appendTrainingMarker(group, trainX, trainY, company.color);
              }}
              if (overlayMode === "release" || overlayMode === "both") {{
                appendReleaseMarker(group, releaseX, releaseY, company.color);
              }}
              rootGroup.appendChild(group);
            }});
          }});
          return;
        }}

        const rowCount = 3;
        const rowGap = 18;
        const companyBandHeight = geometry.modelBandHeight / companies.length;

        companies.forEach((company, companyIndex) => {{
          const models = (modelsByCompany.get(company.name) || []).filter((model) => {{
            const targetTime =
              overlayMode === "train" ? model.training_start_time : model.release_time;
            return targetTime >= minTime && targetTime <= geometry.maxVisibleTime;
          }});
          const laneTop = geometry.margin.top + companyBandHeight * companyIndex;
          const rows = Array.from({{ length: rowCount }}, () => ({{ end: -Infinity }}));
          const items = models
            .map((model) => {{
              const releaseX = clamp(geometry.xForTime(model.release_time), geometry.plotLeft, geometry.plotRight);
              const trainX = clamp(geometry.xForTime(model.training_start_time), geometry.plotLeft, geometry.plotRight);
              const releaseY = geometry.yForValue(interpolateValueAtTime(company.points, model.release_time));
              const trainY = geometry.yForValue(interpolateValueAtTime(company.points, model.training_start_time));
              const chipLabel = model.display_label || model.short_label;
              const chipWidth = measureChipWidth(chipLabel, geometry.isCompact);
              const centerX =
                overlayMode === "both"
                  ? (releaseX + trainX) / 2
                  : overlayMode === "train"
                    ? trainX
                    : releaseX;
              const occupiedStart =
                overlayMode === "both"
                  ? Math.min(trainX, centerX - chipWidth / 2)
                  : centerX - chipWidth / 2;
              const occupiedEnd =
                overlayMode === "both"
                  ? Math.max(releaseX, centerX + chipWidth / 2)
                  : centerX + chipWidth / 2;

              return {{
                ...model,
                releaseX,
                releaseY,
                trainX,
                trainY,
                chipWidth,
                chipLabel,
                centerX,
                occupiedStart,
                occupiedEnd,
              }};
            }})
            .sort((a, b) => a.occupiedStart - b.occupiedStart);

          items.forEach((item) => {{
            const gap = 8;
            let rowIndex = rows.findIndex((row) => item.occupiedStart > row.end + gap);
            if (rowIndex === -1) {{
              rowIndex = rows.reduce((bestIndex, row, index, collection) =>
                row.end < collection[bestIndex].end ? index : bestIndex,
              0);
            }}
            rows[rowIndex].end = item.occupiedEnd;
            item.rowIndex = rowIndex;
          }});

          items.forEach((item) => {{
            const chipHeight = 16;
            const chipY = laneTop + 4 + item.rowIndex * rowGap;
            const chipX = clamp(item.centerX - item.chipWidth / 2, geometry.plotLeft, geometry.plotRight - item.chipWidth);
            const chipCenterX = chipX + item.chipWidth / 2;
            const stemBaseY = chipY + chipHeight + 3;
            const group = makeSvg("g");
            const title = makeSvg("title");
            title.textContent = `${{item.model_name}} | release ${{item.release_date}} | est. train start ${{item.estimated_training_start_date}}`;
            group.appendChild(title);

            if (overlayMode === "both") {{
              group.appendChild(
                makeSvg("line", {{
                  x1: item.trainX,
                  x2: item.trainX,
                  y1: stemBaseY,
                  y2: item.trainY,
                  class: "model-stem",
                  stroke: company.color,
                  "stroke-dasharray": "4 4",
                }})
              );
              group.appendChild(
                makeSvg("line", {{
                  x1: item.releaseX,
                  x2: item.releaseX,
                  y1: stemBaseY,
                  y2: item.releaseY,
                  class: "model-stem",
                  stroke: company.color,
                }})
              );
              group.appendChild(
                makeSvg("line", {{
                  x1: item.trainX,
                  x2: item.releaseX,
                  y1: stemBaseY,
                  y2: stemBaseY,
                  class: "model-window",
                  stroke: company.color,
                  fill: company.color,
                  "stroke-dasharray": "4 4",
                }})
              );
            }} else {{
              const eventX = overlayMode === "train" ? item.trainX : item.releaseX;
              const eventY = overlayMode === "train" ? item.trainY : item.releaseY;
              group.appendChild(
                makeSvg("line", {{
                  x1: eventX,
                  x2: chipCenterX,
                  y1: eventY,
                  y2: stemBaseY,
                  class: "model-stem",
                  stroke: company.color,
                  "stroke-dasharray": overlayMode === "train" ? "4 4" : "",
                }})
              );
            }}

            group.appendChild(
              makeSvg("rect", {{
                x: chipX,
                y: chipY,
                width: item.chipWidth,
                height: chipHeight,
                rx: 8,
                class: "model-chip",
              }})
            );

            const chipText = makeSvg("text", {{
              x: chipCenterX,
              y: chipY + 11,
              "text-anchor": "middle",
              class: "model-chip-text",
              fill: company.color,
            }});
            chipText.textContent = item.chipLabel;
            group.appendChild(chipText);

            if (overlayMode === "both" || overlayMode === "train") {{
              appendTrainingMarker(group, item.trainX, item.trainY, company.color);
            }}
            if (overlayMode === "both" || overlayMode === "release") {{
              appendReleaseMarker(group, item.releaseX, item.releaseY, company.color);
            }}

            rootGroup.appendChild(group);
          }});
        }});
      }}

      function render() {{
        const bounds = stage.getBoundingClientRect();
        const width = Math.max(360, Math.round(bounds.width));
        const height = Math.max(360, Math.round(bounds.height));
        const isCompact = width <= 920;
        const showRightLabels = width >= 980;
        const lineWidth = isCompact ? 4.2 : 5;
        const outerDotRadius = isCompact ? 6.4 : 8;
        const innerDotRadius = isCompact ? 3.8 : 4.5;
        const visibleMaxTime = rangeMode === "to-date" ? currentCutoffTime : maxTime;
        const visibleCompanies = companies.map((company) => ({{
          ...company,
          visiblePoints: company.points.filter((point) => point.time <= visibleMaxTime),
        }}));
        const visibleMaxTotal = Math.max(
          ...visibleCompanies.flatMap((company) =>
            company.visiblePoints.length
              ? company.visiblePoints.map((point) => point.total_gw)
              : [0]
          )
        );
        const yStep = visibleMaxTotal <= 2.5 ? 0.25 : visibleMaxTotal <= 5 ? 0.5 : 1;
        const yMax = Math.ceil((visibleMaxTotal + (rangeMode === "to-date" ? 0.12 : 0.35)) / yStep) * yStep;
        const yTicks = [];
        for (let tick = 0; tick <= yMax + 0.0001; tick += yStep) {{
          yTicks.push(Number(tick.toFixed(2)));
        }}
        const showModelBand = !isCompact && overlayMode !== "off";
        const modelBandHeight = showModelBand ? 124 : 0;
        const margin = isCompact
          ? {{ top: 52, right: 22, bottom: 54, left: 56 }}
          : {{ top: showModelBand ? 18 : 54, right: showRightLabels ? 150 : 24, bottom: 58, left: 66 }};

        svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
        svg.textContent = "";

        const plotTop = margin.top + modelBandHeight;
        const plotBottom = height - margin.bottom;
        const plotLeft = margin.left;
        const plotRight = width - margin.right;
        const plotWidth = plotRight - plotLeft;
        const plotHeight = plotBottom - plotTop;
        const visibleRange = Math.max(1, visibleMaxTime - minTime);
        const xForTime = (time) => plotLeft + ((time - minTime) / visibleRange) * plotWidth;
        const yForValue = (value) => plotBottom - (value / yMax) * plotHeight;
        const linePath = (points, accessor) =>
          points
            .map((point, index) => `${{index === 0 ? "M" : "L"}}${{xForTime(point.time).toFixed(2)}},${{yForValue(accessor(point)).toFixed(2)}}`)
            .join(" ");

        const geometry = {{
          isCompact,
          margin,
          plotTop,
          plotBottom,
          plotLeft,
          plotRight,
          plotWidth,
          plotHeight,
          modelBandHeight,
          maxVisibleTime: visibleMaxTime,
          xForTime,
          yForValue,
        }};

        const rootGroup = makeSvg("g");
        svg.appendChild(rootGroup);

        for (const tick of yTicks) {{
          const y = yForValue(tick);
          rootGroup.appendChild(
            makeSvg("line", {{
              x1: plotLeft,
              x2: plotRight,
              y1: y,
              y2: y,
              class: tick === 0 ? "axis-line" : "grid-line",
            }})
          );

          const label = makeSvg("text", {{
            x: plotLeft - 12,
            y: y + 4,
            "text-anchor": "end",
            class: "tick-label",
          }});
          const tickLabel =
            yStep < 1
              ? `${{tick.toFixed(2).replace(/\\.00$/, "").replace(/0$/, "")}}`
              : `${{tick.toFixed(0)}}`;
          label.textContent = `${{tickLabel}} GW`;
          rootGroup.appendChild(label);
        }}

        for (const item of quarterPositions) {{
          if (item.time > visibleMaxTime) {{
            continue;
          }}
          const x = xForTime(item.time);
          rootGroup.appendChild(
            makeSvg("line", {{
              x1: x,
              x2: x,
              y1: plotTop,
              y2: plotBottom,
              class: item.month.endsWith("-01-31") ? "axis-line" : "quarter-line",
            }})
          );
        }}

        const yearLabelCandidates = [];
        for (const item of yearPositions) {{
          if (item.time > visibleMaxTime) {{
            continue;
          }}
          yearLabelCandidates.push({{
            x: xForTime(item.time),
            text: new Date(`${{item.month}}T00:00:00Z`).getUTCFullYear().toString(),
            anchor: "middle",
            force: false,
          }});
        }}

        if (rangeMode === "to-date") {{
          yearLabelCandidates.push({{
            x: xForTime(visibleMaxTime),
            text: new Date(visibleMaxTime).getUTCFullYear().toString(),
            anchor: "end",
            force: true,
          }});
        }}

        const minYearGap = isCompact ? 56 : 0;
        const filteredYearLabels = [];
        yearLabelCandidates.forEach((candidate) => {{
          if (!isCompact) {{
            filteredYearLabels.push(candidate);
            return;
          }}
          const previous = filteredYearLabels[filteredYearLabels.length - 1];
          if (!previous || candidate.x - previous.x >= minYearGap) {{
            filteredYearLabels.push(candidate);
            return;
          }}
          if (candidate.force) {{
            filteredYearLabels[filteredYearLabels.length - 1] = candidate;
          }}
        }});

        filteredYearLabels.forEach((candidate) => {{
          const label = makeSvg("text", {{
            x: candidate.x,
            y: plotBottom + 24,
            "text-anchor": candidate.anchor,
            class: "year-label",
          }});
          label.textContent = candidate.text;
          rootGroup.appendChild(label);
        }});

        const endLabels = [];

        visibleCompanies.forEach((company) => {{
          const color = company.color;
          const totalPath = makeSvg("path", {{
            d: linePath(company.visiblePoints, (point) => point.total_gw),
            class: "total-line",
            stroke: color,
            "stroke-width": lineWidth,
          }});
          rootGroup.appendChild(totalPath);

          company.visiblePoints
            .filter((point) => point.month_end.endsWith("-12-31"))
            .forEach((point) => {{
              const x = xForTime(point.time);
              const y = yForValue(point.total_gw);
              rootGroup.appendChild(
                makeSvg("circle", {{
                  cx: x,
                  cy: y,
                  r: outerDotRadius,
                  class: "anchor-ring",
                  stroke: color,
                }})
              );
              rootGroup.appendChild(
                makeSvg("circle", {{
                  cx: x,
                  cy: y,
                  r: innerDotRadius,
                  class: "anchor-fill",
                  fill: color,
                }})
              );
            }});

          const lastPoint = company.visiblePoints[company.visiblePoints.length - 1];
          endLabels.push({{
            company,
            color,
            x: plotRight + 16,
            y: yForValue(lastPoint.total_gw),
            text: `${{company.name}} ${{lastPoint.total_gw.toFixed(1)}} GW`,
          }});
        }});

        renderModelOverlay(rootGroup, geometry);

        if (showRightLabels) {{
          endLabels.sort((a, b) => a.y - b.y);
          const minGap = 22;
          for (let index = 1; index < endLabels.length; index += 1) {{
            if (endLabels[index].y - endLabels[index - 1].y < minGap) {{
              endLabels[index].y = endLabels[index - 1].y + minGap;
            }}
          }}
          for (let index = endLabels.length - 2; index >= 0; index -= 1) {{
            if (endLabels[index + 1].y > plotBottom - 4) {{
              endLabels[index + 1].y = plotBottom - 4;
              endLabels[index].y = Math.min(endLabels[index].y, endLabels[index + 1].y - minGap);
            }}
          }}

          endLabels.forEach((item) => {{
            const rightLabel = makeSvg("text", {{
              x: item.x,
              y: Math.max(plotTop + 14, Math.min(plotBottom - 4, item.y)),
              class: "right-label",
              fill: item.color,
            }});
            rightLabel.textContent = item.text;
            rootGroup.appendChild(rightLabel);
          }});
        }}
      }}

      let resizeToken = null;
      const queueRender = () => {{
        if (resizeToken !== null) {{
          cancelAnimationFrame(resizeToken);
        }}
        resizeToken = requestAnimationFrame(() => {{
          resizeToken = null;
          render();
        }});
      }};

      new ResizeObserver(queueRender).observe(stage);
      window.addEventListener("resize", queueRender);
      setOverlayMode(overlayMode);
    </script>
  </body>
</html>
"""


def build_html_payload(
    rows: list[dict[str, object]],
    overlays: list[ModelOverlay],
) -> dict[str, object]:
    by_company: dict[str, list[dict[str, object]]] = {company: [] for company in COMPANIES}
    for row in rows:
        by_company[str(row["company"])].append(
            {
                "month_end": row["month_end"],
                "total_gw": row["total_gw"],
                "floor_gw": row["floor_gw"],
                "uplift_gw": row["uplift_gw"],
                "estimate_basis": row["estimate_basis"],
                "anchor_label": row["anchor_label"],
                "anchor_target_gw": row["anchor_target_gw"],
                "anchor_note": row["anchor_note"],
                "pacing_interval": row["pacing_interval"],
                "floor_basis": row["floor_basis"],
                "floor_events": row["floor_events"].split(" | ") if row["floor_events"] else [],
            }
        )

    company_payload = [
        {
            "key": company.lower(),
            "name": company,
            "color": COMPANY_COLORS[company],
            "points": by_company[company],
        }
        for company in COMPANIES
    ]

    model_payload = [
        {
            "name": company,
            "models": [
                {
                    "model_key": overlay.model_key,
                    "model_name": overlay.model_name,
                    "short_label": overlay.short_label,
                    "display_label": overlay.display_label,
                    "release_date": overlay.release_date.isoformat(),
                    "estimated_training_start_date": overlay.estimated_training_start_date.isoformat(),
                    "source_url": overlay.source_url,
                    "source_title": overlay.source_title,
                }
                for overlay in overlays
                if overlay.company == company
            ],
        }
        for company in COMPANIES
    ]

    return {
        "months": sorted({row["month_end"] for row in rows}),
        "companies": company_payload,
        "models": model_payload,
        "training_lag_months": MODEL_TRAINING_LAG_MONTHS,
        "default_overlay_mode": "release",
        "default_range_mode": "full",
        "current_cutoff_date": DATE_BY_ROW_KEY["2026_current"].isoformat(),
    }


def main() -> None:
    paths.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    paths.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    monthly_rows = build_monthly_rows()
    model_overlays = read_model_overlays()
    write_monthly_csv(monthly_rows)
    write_model_overlay_csv(model_overlays)
    MONTHLY_HTML.write_text(
        html_template(build_html_payload(monthly_rows, model_overlays)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
