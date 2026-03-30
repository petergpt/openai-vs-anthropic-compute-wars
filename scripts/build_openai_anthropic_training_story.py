from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

import build_openai_anthropic_monthly_visualization as monthly
import paths

STORY_HTML = paths.PROJECT_ROOT / "outputs" / "openai_anthropic_training_story.html"
DOCS_STORY_HTML = paths.TRAINING_STORY_HTML
DOCS_INDEX_HTML = paths.DOCS_INDEX_HTML
OPEN_DATA_DERIVED = paths.PACKAGE_DIR / "derived"
EVIDENCE_PACK_CSV = paths.OPENAI_ANTHROPIC_EVIDENCE_PACK_CSV
EXCLUDED_MODEL_KEYS = {"openai:gpt-4.1"}
EPOCH_DATA_CENTERS_URL = "https://epoch.ai/data/data-centers"
EPOCH_DATA_CENTERS_LABEL = "Epoch AI data centers"
CANONICAL_SOURCE_URLS = {
    "https://apnews.com/article/275df9e291bbb361962dfc6ba868506c": "https://apnews.com/article/anthropic-google-275df9e291bbb361962dfc6ba868506c",
    "https://gis.maricopa.gov/aqd/recordsviewer/#": "https://gis.maricopa.gov/aqd/recordsviewer/",
    "https://www.anthropic.com/news/trainium2-and-distillation": "https://claude.com/blog/trainium2-and-distillation",
    "https://youtu.be/hobvps-H38o?si=wsAcRtDFG4Df4_7z&t=901": "https://www.youtube.com/watch?si=wsAcRtDFG4Df4_7z&t=901&v=hobvps-H38o&feature=youtu.be",
}
DATA_EXPLORER_DATASETS = [
    {
        "key": "core_data",
        "title": "Core data",
        "description": "",
    },
]


def iso_to_ordinal(value: str) -> int:
    return date.fromisoformat(value).toordinal()


def interpolate_value(points: list[dict[str, object]], target_iso: str) -> float:
    target = iso_to_ordinal(target_iso)
    if target <= iso_to_ordinal(str(points[0]["month_end"])):
        return float(points[0]["total_gw"])
    if target >= iso_to_ordinal(str(points[-1]["month_end"])):
        return float(points[-1]["total_gw"])

    for index in range(1, len(points)):
        current = points[index]
        current_time = iso_to_ordinal(str(current["month_end"]))
        if target <= current_time:
            previous = points[index - 1]
            previous_time = iso_to_ordinal(str(previous["month_end"]))
            span = current_time - previous_time
            progress = 0.0 if span == 0 else (target - previous_time) / span
            previous_value = float(previous["total_gw"])
            current_value = float(current["total_gw"])
            return previous_value + (current_value - previous_value) * progress

    return float(points[-1]["total_gw"])


def trim_company_prefix(model_name: str, company: str) -> str:
    if company == "OpenAI" and model_name.startswith("OpenAI "):
        return model_name.removeprefix("OpenAI ")
    return model_name


def story_label(model_key: str, model_name: str, company: str) -> str:
    if company == "OpenAI":
        return trim_company_prefix(model_name, company)

    anthropic_labels = {
        "anthropic:claude-2": "Claude 2",
        "anthropic:claude-2.1": "Claude 2.1",
    }
    if model_key in anthropic_labels:
        return anthropic_labels[model_key]

    tail = model_key.removeprefix("anthropic:claude-")
    match = re.fullmatch(
        r"(?:(?P<version_first>\d+(?:\.\d+)*)-(?P<family_last>opus|sonnet)|(?P<family_first>opus|sonnet)-(?P<version_last>\d+(?:\.\d+)*))(?:-(?P<suffix>new))?",
        tail,
    )
    if match:
        family = match.group("family_first") or match.group("family_last")
        version = match.group("version_first") or match.group("version_last")
        suffix = match.group("suffix")
        label = f"{family.capitalize()} {version}"
        if suffix:
            label += f" {suffix.capitalize()}"
        return label

    return model_name.removeprefix("Claude ")


CONFIDENCE_TAG_RE = re.compile(r"\s+#(?:confident|likely|possible|speculative)\b", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def clean_display_text(value: str) -> str:
    if not value:
        return ""
    without_tags = CONFIDENCE_TAG_RE.sub("", value)
    without_links = MARKDOWN_LINK_RE.sub(r"\1", without_tags)
    return re.sub(r"\s+", " ", without_links).strip()


def read_packaged_csv(filename: str) -> tuple[list[str], list[dict[str, str]]]:
    path = OPEN_DATA_DERIVED / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_file_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "file://")):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.as_uri()
    return value


def is_local_source_href(href: str) -> bool:
    normalized = maybe_file_uri(href.strip())
    return bool(normalized) and not normalized.startswith(("http://", "https://"))


def normalize_source_link(link: dict[str, str]) -> dict[str, str]:
    href = maybe_file_uri(link.get("href", "").strip())
    label = link.get("label", href).strip() or href
    if href and not href.startswith(("http://", "https://")):
        return {
            "label": EPOCH_DATA_CENTERS_LABEL,
            "href": EPOCH_DATA_CENTERS_URL,
        }
    href = CANONICAL_SOURCE_URLS.get(href, href)
    return {
        "label": label,
        "href": href,
    }


def inferred_company_label(primary_user: str, users_raw: str) -> str:
    if primary_user in {"OpenAI", "Anthropic"}:
        return primary_user
    normalized_users = users_raw.lower()
    mentions_openai = "openai" in normalized_users
    mentions_anthropic = "anthropic" in normalized_users
    if mentions_anthropic and not mentions_openai:
        return "Anthropic"
    if mentions_openai and not mentions_anthropic:
        return "OpenAI"
    return ""


def build_link_cell(links: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in links:
        normalized = normalize_source_link(link)
        href = normalized["href"].strip()
        if not href or href in seen:
            continue
        seen.add(href)
        trimmed.append(normalized)
        if len(trimmed) >= limit:
            break
    return trimmed


def float_or_zero(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


ESTIMATE_TYPE_LABELS = {
    "official_exact_self_reported_available_compute": "Official company total",
    "triangulated_range_above_site_floor": "Triangulated above visible floor",
    "triangulated_range_with_floor_low": "Triangulated with floor lower bound",
    "heuristic_site_backed_lower_bound_only": "Conservative site-backed floor proxy",
    "floor_plus_platform_uplift_range": "Visible floor plus platform uplift",
    "synthetic_baseline": "Synthetic baseline",
}

POINT_ESTIMATE_TYPE_LABELS = {
    "historical_exact_or_internal_center_of_gravity": "Historical point estimate",
    "future_single_estimate_from_debated_range": "Chosen point within debated range",
    "conservative_forward_estimate_beyond_high_confidence_window": "Conservative forward point",
    "synthetic_baseline": "Synthetic baseline",
}

ROLE_LABELS = {
    "anchor": "Anchor",
    "uplift": "Uplift",
    "envelope": "Envelope",
    "lower_bound": "Lower bound",
    "continuity_proof": "Continuity proof",
}


def prettify_token(value: str, mapping: dict[str, str] | None = None) -> str:
    if not value:
        return ""
    if mapping and value in mapping:
        return mapping[value]
    words = []
    for part in value.split("_"):
        if part.upper() in {"AI", "AWS", "TPU", "GPU"}:
            words.append(part.upper())
        elif part.isupper():
            words.append(part)
        else:
            words.append(part.capitalize())
    return " ".join(words)


def build_data_explorer_payload() -> dict[str, object]:
    _, anchors = read_packaged_csv("yearly_anchor_registry.csv")
    _, site_floor_rows = read_packaged_csv("site_floor_components_monthly.csv")
    _, data_centers = read_packaged_csv("data_center_registry.csv")
    _, timeline_sources = read_packaged_csv("data_center_timeline_row_sources.csv")
    _, selected_sources = read_packaged_csv("data_center_selected_sources.csv")
    _, anchor_evidence_links = read_packaged_csv("anchor_evidence_links.csv")
    evidence_pack = read_csv_rows(EVIDENCE_PACK_CSV)

    timeline_sources_by_row: dict[str, list[dict[str, str]]] = {}
    for row in timeline_sources:
        timeline_sources_by_row.setdefault(row["timeline_row_id"], []).append(row)

    selected_sources_by_center: dict[str, list[dict[str, str]]] = {}
    for row in selected_sources:
        selected_sources_by_center.setdefault(row["data_center_id"], []).append(row)

    evidence_by_id = {
        row["evidence_id"]: row
        for row in evidence_pack
    }
    anchor_evidence_by_anchor: dict[str, list[dict[str, str]]] = {}
    for row in anchor_evidence_links:
        anchor_evidence_by_anchor.setdefault(row["anchor_id"], []).append(row)

    anchor_label_by_id = {
        row["anchor_id"]: row["anchor_label"]
        for row in anchors
    }

    current_month = monthly.DATE_BY_ROW_KEY["2026_current"].isoformat()

    site_floor_by_point: dict[tuple[str, str], list[dict[str, str]]] = {}
    site_floor_by_center: dict[str, list[dict[str, str]]] = {}
    center_company: dict[str, str] = {}
    for row in site_floor_rows:
        site_floor_by_point.setdefault((row["company"], row["month_end"]), []).append(row)
        site_floor_by_center.setdefault(row["data_center_id"], []).append(row)
        center_company[row["data_center_id"]] = row["company"]
    for rows in site_floor_by_center.values():
        rows.sort(key=lambda row: row["month_end"])

    data_centers_by_id = {row["data_center_id"]: row for row in data_centers}
    for row in data_centers:
        center_company.setdefault(
            row["data_center_id"],
            inferred_company_label(row["primary_user"], row["users_raw"]),
        )

    data_centers_by_company: dict[str, list[dict[str, str]]] = {company: [] for company in monthly.COMPANIES}
    for row in data_centers:
        company = center_company.get(row["data_center_id"], "")
        if company in data_centers_by_company:
            data_centers_by_company[company].append(row)
    for rows in data_centers_by_company.values():
        rows.sort(key=lambda row: row["data_center"])

    def center_sources(center_id: str) -> list[dict[str, str]]:
        sources = [
            {
                "label": source_row["source_title"],
                "href": source_row["source_url"],
            }
            for source_row in selected_sources_by_center.get(center_id, [])
            if source_row["source_kind"] != "calculations_sheet"
        ]
        if not sources:
            sources = [
                {
                    "label": source_row["source_title"],
                    "href": source_row["source_url"],
                }
                for source_row in selected_sources_by_center.get(center_id, [])
            ]
        return build_link_cell(sources)

    def row_sources(component_row: dict[str, str]) -> list[dict[str, str]]:
        sources = [
            {
                "label": source_row["source_title"],
                "href": source_row["source_url"],
            }
            for source_row in timeline_sources_by_row.get(component_row["timeline_row_id"], [])
        ]
        if not sources:
            sources = center_sources(component_row["data_center_id"])
        return build_link_cell(sources)

    point_views: list[dict[str, object]] = []
    download_rows: list[dict[str, object]] = []

    for row in anchors:
        if row["row_key"].startswith("synthetic_"):
            continue

        company = row["company"]
        anchor_date = row["anchor_date"]
        total_gw = float_or_zero(row["chart_effective_anchor_gw"])
        floor_gw = float_or_zero(row["site_floor_gw"])
        uplift_gw = float_or_zero(row["point_minus_floor_gw"])

        anchor_sources: list[dict[str, str]] = []
        anchor_evidence_items: list[dict[str, object]] = []
        for link_row in anchor_evidence_by_anchor.get(row["anchor_id"], []):
            evidence = evidence_by_id.get(link_row["evidence_id"], {})
            href = str(evidence.get("url", "")).strip()
            if not href:
                continue
            normalized_link = normalize_source_link(
                {"label": evidence.get("title") or link_row["evidence_id"], "href": href}
            )
            local_source = is_local_source_href(href)
            anchor_sources.append(normalized_link)
            anchor_evidence_items.append(
                {
                    "title": normalized_link["label"],
                    "href": normalized_link["href"],
                    "source_family": "Epoch AI dataset"
                    if local_source
                    else prettify_token(str(evidence.get("source_family", ""))),
                    "evidence_type": prettify_token(str(evidence.get("evidence_type", ""))),
                    "role": prettify_token(str(evidence.get("role_in_model", "")), ROLE_LABELS),
                    "quant_signal": str(evidence.get("quant_signal", "")).strip(),
                    "note": str(evidence.get("note", "")).strip() or str(link_row.get("note", "")).strip(),
                }
            )

        component_rows = sorted(
            site_floor_by_point.get((company, anchor_date), []),
            key=lambda item: (-float_or_zero(item["allocated_power_gw"]), item["data_center"]),
        )
        floor_components: list[dict[str, object]] = []
        component_ids: set[str] = set()
        for component in component_rows:
            component_ids.add(component["data_center_id"])
            registry_row = data_centers_by_id.get(component["data_center_id"], {})
            floor_components.append(
                {
                    "data_center_id": component["data_center_id"],
                    "data_center": component["data_center"],
                    "allocated_power_gw": round(float_or_zero(component["allocated_power_gw"]), 3),
                    "allocated_h100_equivalents": int(round(float_or_zero(component["allocated_h100_equivalents"]))),
                    "operational_buildings": int(round(float_or_zero(component["timeline_buildings_operational"]))),
                    "timeline_status": clean_display_text(component["timeline_status"]),
                    "project": clean_display_text(str(registry_row.get("project", ""))),
                    "country": registry_row.get("country", ""),
                    "sources": row_sources(component),
                }
            )

        future_sites: list[dict[str, object]] = []
        near_term_limit = iso_to_ordinal(anchor_date) + 550
        for center in data_centers_by_company.get(company, []):
            center_id = center["data_center_id"]
            if center_id in component_ids:
                continue
            future_rows = [
                item
                for item in site_floor_by_center.get(center_id, [])
                if item["company"] == company
                and item["month_end"] > anchor_date
                and float_or_zero(item["allocated_power_gw"]) > 0
                and iso_to_ordinal(item["month_end"]) <= near_term_limit
            ]
            if not future_rows:
                continue
            first_future = future_rows[0]
            future_sites.append(
                {
                    "data_center_id": center_id,
                    "data_center": center["data_center"],
                    "first_live_date": first_future["month_end"],
                    "first_live_power_gw": round(float_or_zero(first_future["allocated_power_gw"]), 3),
                    "project": clean_display_text(str(center.get("project", ""))),
                    "country": center.get("country", ""),
                    "note": clean_display_text(first_future["timeline_status"]),
                    "sources": row_sources(first_future),
                }
            )
        future_sites.sort(key=lambda item: (str(item["first_live_date"]), str(item["data_center"])))

        floor_source_links = build_link_cell(
            [link for component in floor_components for link in component["sources"]],
            limit=8,
        )
        anchor_source_links = build_link_cell(anchor_sources, limit=8)
        point_id = row["anchor_id"]
        point_payload = {
            "point_id": point_id,
            "company": company,
            "period": row["anchor_label"],
            "date": anchor_date,
            "total_gw": round(total_gw, 3),
            "floor_gw": round(floor_gw, 3),
            "uplift_gw": round(uplift_gw, 3),
            "range_low_gw": round(float_or_zero(row["low_gw"]), 3) if str(row["low_gw"]).strip() else None,
            "range_base_gw": round(float_or_zero(row["base_gw"]), 3) if str(row["base_gw"]).strip() else None,
            "range_high_gw": round(float_or_zero(row["high_gw"]), 3) if str(row["high_gw"]).strip() else None,
            "summary_note": row["summary_note"],
            "estimate_basis": prettify_token(row["company_estimate_type"], ESTIMATE_TYPE_LABELS),
            "point_choice": prettify_token(row["point_estimate_type"], POINT_ESTIMATE_TYPE_LABELS),
            "assumption_ids": row["assumption_ids"],
            "floor_components": floor_components,
            "future_sites": future_sites,
            "floor_source_links": floor_source_links,
            "uplift_source_links": anchor_source_links,
            "uplift_evidence": anchor_evidence_items,
            "component_count": len(floor_components),
        }
        point_views.append(point_payload)

        source_text = build_link_cell(anchor_sources, limit=8)
        year_label = anchor_date[:4]
        sort_order = 0

        def append_download_row(
            *,
            group: str,
            row_type: str,
            label: str,
            gw: float | None,
            detail: str,
            note: str,
            sources: list[dict[str, str]],
        ) -> None:
            nonlocal sort_order
            download_rows.append(
                {
                    "company": company,
                    "year": year_label,
                    "period": row["anchor_label"],
                    "anchor_date": anchor_date,
                    "group": group,
                    "row_type": row_type,
                    "label": label,
                    "gw": round(gw, 3) if gw is not None else "",
                    "detail": detail,
                    "note": note,
                    "sources": sources,
                    "sort_order": sort_order,
                }
            )
            sort_order += 1

        append_download_row(
            group="summary",
            row_type="total",
            label=row["anchor_label"],
            gw=total_gw,
            detail=" | ".join(
                part
                for part in [
                    prettify_token(row["company_estimate_type"], ESTIMATE_TYPE_LABELS),
                    prettify_token(row["point_estimate_type"], POINT_ESTIMATE_TYPE_LABELS),
                ]
                if part
            ),
            note=row["summary_note"],
            sources=source_text,
        )
        append_download_row(
            group="named_sites",
            row_type="subtotal",
            label="Named sites subtotal",
            gw=floor_gw,
            detail=f"{len(floor_components)} named site(s)",
            note="Sum of named live sites at this point.",
            sources=floor_source_links,
        )
        append_download_row(
            group="other_compute",
            row_type="subtotal",
            label="Other compute subtotal",
            gw=uplift_gw,
            detail=prettify_token(row["point_estimate_type"], POINT_ESTIMATE_TYPE_LABELS),
            note="Compute supported beyond the named live sites.",
            sources=anchor_source_links,
        )
        for component in floor_components:
            append_download_row(
                group="named_sites",
                row_type="site",
                label=component["data_center"],
                gw=component["allocated_power_gw"],
                detail=f'{component["operational_buildings"]} building(s) | {component["allocated_h100_equivalents"]} H100e',
                note=component["timeline_status"],
                sources=component["sources"],
            )
        for evidence_item in anchor_evidence_items:
            append_download_row(
                group="other_compute",
                row_type="evidence",
                label=str(evidence_item["title"]),
                gw=None,
                detail=" | ".join(
                    bit
                    for bit in [
                        str(evidence_item.get("role", "")).strip(),
                        str(evidence_item.get("evidence_type", "")).strip(),
                    ]
                    if bit
                ),
                note=str(evidence_item.get("quant_signal", "")).strip()
                or str(evidence_item.get("note", "")).strip(),
                sources=build_link_cell(
                    [{"label": str(evidence_item["title"]), "href": str(evidence_item["href"])}],
                    limit=1,
                ),
            )
        for future_site in future_sites:
            append_download_row(
                group="future_sites",
                row_type="site",
                label=future_site["data_center"],
                gw=future_site["first_live_power_gw"],
                detail=f'0 now | first live {future_site["first_live_date"]}',
                note=future_site["note"],
                sources=future_site["sources"],
            )

    point_views.sort(key=lambda item: (str(item["company"]), str(item["date"])))
    download_rows.sort(
        key=lambda item: (
            str(item["year"]),
            str(item["anchor_date"]),
            str(item["company"]),
            int(item["sort_order"]),
            str(item["label"]),
        )
    )

    return {
        "package_dir": EPOCH_DATA_CENTERS_URL,
        "default_company": "Both",
        "points": point_views,
        "download": {
            "filename": "anthropic-openai-compute-breakdown.csv",
            "columns": [
                "company",
                "year",
                "period",
                "anchor_date",
                "group",
                "row_type",
                "label",
                "gw",
                "detail",
                "note",
                "sources",
                "sort_order",
            ],
            "rows": download_rows,
        },
    }


def build_story_payload(
    rows: list[dict[str, object]],
    overlays: list[monthly.ModelOverlay],
) -> dict[str, object]:
    cutoff_date = monthly.DATE_BY_ROW_KEY["2026_current"]
    cutoff_iso = cutoff_date.isoformat()
    projection_end_iso = max(str(row["month_end"]) for row in rows)

    by_company: dict[str, list[dict[str, object]]] = {company: [] for company in monthly.COMPANIES}
    for row in rows:
        company = str(row["company"])
        by_company[company].append(
            {
                "month_end": str(row["month_end"]),
                "total_gw": float(row["total_gw"]),
            }
        )

    company_payload: list[dict[str, object]] = []
    for company in monthly.COMPANIES:
        points = by_company[company]
        company_models = []
        for overlay in overlays:
            if (
                overlay.company != company
                or overlay.release_date > cutoff_date
                or overlay.model_key in EXCLUDED_MODEL_KEYS
            ):
                continue
            training_start_iso = overlay.estimated_training_start_date.isoformat()
            release_iso = overlay.release_date.isoformat()
            company_models.append(
                {
                    "model_key": overlay.model_key,
                    "model_name": overlay.model_name,
                    "label_name": trim_company_prefix(overlay.model_name, company),
                    "story_label": story_label(overlay.model_key, overlay.model_name, company),
                    "display_label": overlay.display_label,
                    "training_start_date": training_start_iso,
                    "training_start_gw": round(
                        interpolate_value(points, training_start_iso),
                        6,
                    ),
                    "release_date": release_iso,
                    "release_gw": round(
                        interpolate_value(points, release_iso),
                        6,
                    ),
                    "source_url": overlay.source_url,
                    "source_title": overlay.source_title,
                }
            )

        company_payload.append(
            {
                "key": company.lower(),
                "name": company,
                "color": monthly.COMPANY_COLORS[company],
                "points": points,
                "models": company_models,
            }
        )

    return {
        "title": "OpenAI vs Anthropic Compute Wars",
        "subtitle": "Source: Best-effort estimate from Epoch AI Frontier Data Centers and selected public disclosures.",
        "legend": {
            "training": "Estimated training start",
            "release": "Release point",
            "window": "Active training window",
        },
        "companies": company_payload,
        "story_start_date": monthly.START_MONTH.isoformat(),
        "story_end_date": projection_end_iso,
        "current_cutoff_date": cutoff_iso,
        "training_lag_months": monthly.MODEL_TRAINING_LAG_MONTHS,
        "data_explorer": build_data_explorer_payload(),
    }


def html_template(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, separators=(",", ":"))
    template = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>OpenAI vs Anthropic Compute Wars</title>
    <style>
      :root {
        --page: #f5f2eb;
        --surface: rgba(255, 255, 255, 0.84);
        --surface-strong: rgba(255, 255, 255, 0.96);
        --ink: #12110f;
        --muted: #625b53;
        --line: rgba(18, 17, 15, 0.12);
        --grid: rgba(18, 17, 15, 0.08);
        --openai: #101010;
        --anthropic: #db5c2b;
        --body-font: "Avenir Next", "Helvetica Neue", sans-serif;
        --display-font: "Helvetica Neue", "Avenir Next Condensed", "Arial Narrow", sans-serif;
        --hero-date-font: var(--display-font);
        --body-background: linear-gradient(180deg, #f7f5f0 0%, #ece9e2 100%);
        --body-grid: linear-gradient(rgba(18, 17, 15, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(18, 17, 15, 0.02) 1px, transparent 1px);
        --body-grid-size: 44px;
        --body-grid-opacity: 1;
        --body-grid-mask: linear-gradient(to bottom, rgba(0, 0, 0, 0.55), transparent 80%);
        --poster-background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 245, 239, 0.95));
        --poster-border: rgba(18, 17, 15, 0.08);
        --poster-radius: 16px;
        --poster-shadow: 0 10px 36px rgba(18, 17, 15, 0.08);
        --poster-blur: none;
        --poster-grid: linear-gradient(rgba(18, 17, 15, 0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(18, 17, 15, 0.035) 1px, transparent 1px);
        --poster-grid-size: 120px;
        --poster-grid-opacity: 1;
        --poster-grid-mask: linear-gradient(to right, rgba(0, 0, 0, 0.28), transparent 84%);
        --title-color: var(--ink);
        --title-weight: 780;
        --title-tracking: -0.06em;
        --title-transform: none;
        --legend-size: 1.16rem;
        --legend-weight: 800;
        --legend-line-width: 44px;
        --legend-line-thickness: 5px;
        --control-radius: 10px;
        --control-bg: rgba(255, 255, 255, 0.92);
        --control-border: rgba(18, 17, 15, 0.16);
        --control-text: rgba(18, 17, 15, 0.76);
        --control-active-bg: #12110f;
        --control-active-border: #12110f;
        --control-active-text: rgba(255, 250, 245, 0.97);
        --control-shadow: 0 4px 12px rgba(18, 17, 15, 0.1);
        --control-transform: uppercase;
        --control-spacing: 0.08em;
        --timeline-accent: #12110f;
        --table-radius: 12px;
        --data-wrap-bg: rgba(255, 255, 255, 0.82);
        --data-wrap-border: rgba(18, 17, 15, 0.08);
        --data-head-bg: rgba(247, 243, 235, 0.96);
        --data-row-alt: rgba(18, 17, 15, 0.018);
        --data-link-bg: rgba(49, 77, 99, 0.08);
        --data-link-border: rgba(49, 77, 99, 0.22);
        --data-link-text: #314d63;
        --data-link-hover-bg: rgba(49, 77, 99, 0.14);
        --data-link-hover-border: rgba(49, 77, 99, 0.32);
        --axis-line-color: rgba(18, 17, 15, 0.16);
        --quarter-line-color: rgba(18, 17, 15, 0.05);
        --projection-region-fill: rgba(18, 17, 15, 0.025);
        --projection-hatch-stroke: rgba(219, 92, 43, 0.12);
        --current-divider-color: rgba(18, 17, 15, 0.22);
        --playhead-line-color: rgba(18, 17, 15, 0.18);
        --marker-stroke: rgba(255, 255, 255, 0.96);
        --label-box-fill: rgba(255, 255, 255, 0.96);
        --hero-date-fill: rgba(18, 17, 15, 0.08);
        --playhead-chip-fill: #12110f;
        --playhead-chip-text: #ffffff;
        --head-halo-opacity: 0.16;
        --reveal-glow-opacity: 0.1;
        --skeleton-opacity: 0.1;
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        margin: 0;
        min-height: 100%;
      }

      body {
        min-height: 100vh;
        min-height: 100svh;
        min-height: 100dvh;
        overflow: hidden;
        font-family: var(--body-font);
        color: var(--ink);
        background: var(--body-background);
      }

      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image: var(--body-grid);
        background-size: var(--body-grid-size) var(--body-grid-size);
        opacity: var(--body-grid-opacity);
        mask-image: var(--body-grid-mask);
      }

      .story {
        display: grid;
        grid-template-rows: minmax(0, 1fr) auto;
        gap: 0.6rem;
        height: 100vh;
        height: 100svh;
        height: 100dvh;
        padding: 1rem;
        overflow: hidden;
      }

      .poster {
        position: relative;
        display: grid;
        grid-template-rows: auto auto 1fr;
        min-height: 0;
        height: 100%;
        border: 1px solid var(--poster-border);
        border-radius: var(--poster-radius);
        overflow: hidden;
        background: var(--poster-background);
        box-shadow: var(--poster-shadow);
        backdrop-filter: var(--poster-blur);
      }

      .poster::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: var(--poster-grid);
        background-size: var(--poster-grid-size) var(--poster-grid-size);
        opacity: var(--poster-grid-opacity);
        mask-image: var(--poster-grid-mask);
      }

      .masthead {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 0.5rem 1rem;
        align-items: start;
        padding: 0.9rem 1.25rem 0.25rem;
      }

      .title-block {
        display: grid;
        gap: 0.18rem;
        min-width: 0;
      }

      h1 {
        margin: 0;
        font-family: var(--display-font);
        font-size: clamp(1.45rem, 3vw, 3.45rem);
        line-height: 0.92;
        letter-spacing: var(--title-tracking);
        color: var(--title-color);
        font-weight: var(--title-weight);
        text-transform: var(--title-transform);
        white-space: nowrap;
      }

      .subtitle {
        margin: 0;
        max-width: 54rem;
        color: var(--muted);
        font-size: 0.7rem;
        line-height: 1.2;
        letter-spacing: 0.015em;
      }

      .legend {
        display: flex;
        flex-wrap: nowrap;
        align-self: center;
        gap: 1.05rem 1.8rem;
        margin-top: 0.18rem;
        font-size: var(--legend-size);
        font-weight: var(--legend-weight);
        letter-spacing: 0.01em;
      }

      .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.78rem;
      }

      .legend-line,
      .legend-window,
      .legend-diamond,
      .legend-circle {
        flex: 0 0 auto;
      }

      .legend-line {
        width: var(--legend-line-width);
        height: 0;
        border-top: var(--legend-line-thickness) solid currentColor;
      }

      .legend-window {
        width: 22px;
        height: 0;
        border-top: 8px solid currentColor;
        opacity: 0.2;
        border-radius: 999px;
      }

      .legend-diamond,
      .legend-circle {
        width: 10px;
        height: 10px;
        border: 2px solid currentColor;
        background: rgba(255, 255, 255, 0.92);
      }

      .legend-diamond {
        transform: rotate(45deg);
      }

      .legend-circle {
        border-radius: 999px;
        background: currentColor;
      }

      .tab-strip {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.2rem 1.25rem 0.35rem;
      }

      .tab-button {
        border: 1px solid var(--control-border);
        background: var(--control-bg);
        color: var(--control-text);
        border-radius: var(--control-radius);
        min-width: 78px;
        padding: 0.4rem 0.78rem 0.35rem;
        font-family: var(--body-font);
        font-size: 0.74rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: var(--control-spacing);
        text-transform: var(--control-transform);
        cursor: pointer;
        transition: background 180ms ease, color 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
      }

      .tab-button[aria-selected="true"] {
        background: var(--control-active-bg);
        color: var(--control-active-text);
        border-color: var(--control-active-border);
        box-shadow: var(--control-shadow);
      }

      .panel-stack {
        position: relative;
        min-height: 0;
        height: 100%;
      }

      .panel-hidden {
        display: none !important;
      }

      .playback-controls,
      .timeline-controls,
      .speed-controls,
      .theme-controls {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        flex: 0 0 auto;
        white-space: nowrap;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: var(--control-spacing);
        text-transform: var(--control-transform);
      }

      .playback-controls {
        gap: 0.32rem;
      }

      .timeline-controls {
        gap: 0.32rem;
      }

      .view-controls {
        display: inline-flex;
        align-items: center;
        gap: 0.34rem;
        flex: 0 0 auto;
        white-space: nowrap;
        color: var(--muted);
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: var(--control-spacing);
        text-transform: var(--control-transform);
      }

      .control-label {
        margin-right: 0.08rem;
      }

      .control-button,
      .view-toggle {
        border: 1px solid var(--control-border);
        background: var(--control-bg);
        color: var(--control-text);
        border-radius: var(--control-radius);
        min-width: 78px;
        padding: 0.34rem 0.58rem 0.31rem;
        font-family: var(--body-font);
        font-size: inherit;
        line-height: 1;
        font-weight: 800;
        letter-spacing: var(--control-spacing);
        text-transform: var(--control-transform);
        cursor: pointer;
        transition: background 180ms ease, color 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
      }

      .control-button {
        min-width: 72px;
      }

      .control-button-primary {
        background: var(--control-active-bg);
        color: var(--control-active-text);
        border-color: var(--control-active-border);
        box-shadow: var(--control-shadow);
      }

      .control-button[aria-pressed="true"],
      .view-toggle[aria-pressed="true"] {
        background: var(--control-active-bg);
        color: var(--control-active-text);
        border-color: var(--control-active-border);
        box-shadow: var(--control-shadow);
      }

      .story-controls {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        gap: 0.38rem;
        flex-wrap: nowrap;
        min-height: 2.3rem;
        padding: 0 0.15rem 0.1rem;
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-width: none;
        -ms-overflow-style: none;
        scroll-behavior: smooth;
      }

      .story-controls::-webkit-scrollbar {
        display: none;
      }

      .speed-label {
        margin-right: 0.08rem;
      }

      .speed-presets {
        display: inline-flex;
        align-items: center;
        gap: 0.18rem;
        flex: 0 0 auto;
        white-space: nowrap;
      }

      .theme-presets {
        display: inline-flex;
        align-items: center;
        gap: 0.16rem;
        flex: 0 0 auto;
        white-space: nowrap;
      }

      .speed-preset,
      .theme-preset {
        border: 1px solid var(--control-border);
        background: var(--control-bg);
        color: var(--control-text);
        border-radius: var(--control-radius);
        min-width: 44px;
        padding: 0.28rem 0.36rem 0.25rem;
        font-family: var(--body-font);
        font-size: 0.62rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: var(--control-spacing);
        text-transform: var(--control-transform);
        cursor: pointer;
        transition: background 180ms ease, color 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
      }

      .theme-preset {
        min-width: 0;
      }

      .speed-preset[aria-pressed="true"],
      .theme-preset[aria-pressed="true"] {
        background: var(--control-active-bg);
        color: var(--control-active-text);
        border-color: var(--control-active-border);
        box-shadow: var(--control-shadow);
      }

      .timeline-range {
        width: 158px;
        accent-color: var(--timeline-accent);
      }

      .timeline-value {
        min-width: 3.8rem;
        font-size: 0.6rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-align: right;
      }

      .stage-shell {
        position: relative;
        min-height: 0;
        height: 100%;
        padding: 0 0.65rem 0.65rem;
      }

      #chart {
        display: block;
        width: 100%;
        height: 100%;
      }

      .data-shell {
        display: grid;
        grid-template-rows: auto auto 1fr;
        gap: 0.72rem;
        min-height: 0;
        height: 100%;
        padding: 0 1.1rem 1rem;
      }

      .data-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        flex-wrap: wrap;
      }

      .data-company-filters {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        flex-wrap: wrap;
      }

      .data-toolbar-right {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .data-pill,
      .data-download {
        border: 1px solid var(--control-border);
        background: var(--control-bg);
        color: var(--control-text);
        border-radius: var(--control-radius);
        padding: 0.34rem 0.64rem 0.31rem;
        font-family: var(--body-font);
        font-size: 0.68rem;
        line-height: 1;
        font-weight: 700;
        letter-spacing: var(--control-spacing);
        text-transform: var(--control-transform);
        cursor: pointer;
      }

      .data-pill[aria-pressed="true"] {
        background: var(--control-active-bg);
        color: var(--control-active-text);
        border-color: var(--control-active-border);
      }

      .data-download {
        color: var(--control-active-text);
        background: var(--control-active-bg);
        border-color: var(--control-active-border);
        box-shadow: var(--control-shadow);
      }

      .data-meta {
        color: var(--muted);
        font-size: 0.78rem;
        letter-spacing: 0.02em;
      }

      .data-link-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 0.28rem 0.42rem;
      }

      .data-inspector-wrap {
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid var(--data-wrap-border);
        border-radius: var(--table-radius);
        background: var(--data-wrap-bg);
        padding: 0.85rem 0.9rem 1rem;
      }

      .data-inspector {
        display: grid;
        gap: 1.05rem;
      }

      .data-year-group {
        display: grid;
        grid-template-columns: 88px minmax(0, 1fr);
        gap: 0.95rem;
        align-items: start;
      }

      .data-year-label {
        margin: 0;
        padding-top: 0.06rem;
        color: rgba(18, 17, 15, 0.2);
        font-family: var(--display-font);
        font-size: clamp(1.6rem, 2vw, 2.4rem);
        line-height: 0.92;
        letter-spacing: -0.06em;
        font-weight: 760;
      }

      .data-year-thread {
        position: relative;
        display: grid;
        gap: 1rem;
        padding-left: 1rem;
      }

      .data-year-thread::before {
        content: "";
        position: absolute;
        left: 0.14rem;
        top: 0.18rem;
        bottom: 0.24rem;
        width: 1px;
        background: rgba(18, 17, 15, 0.08);
      }

      .data-point-flow {
        position: relative;
        display: grid;
        gap: 0.74rem;
        padding-bottom: 0.12rem;
      }

      .data-point-flow::before {
        content: "";
        position: absolute;
        left: -0.98rem;
        top: 0.36rem;
        width: 0.46rem;
        height: 0.46rem;
        border-radius: 999px;
        background: var(--surface-strong);
        border: 1px solid rgba(18, 17, 15, 0.16);
      }

      .data-point-flow + .data-point-flow::after {
        content: "";
        position: absolute;
        left: -0.75rem;
        top: -0.52rem;
        width: 0.3rem;
        height: 1px;
        background: rgba(18, 17, 15, 0.08);
      }

      .data-summary-card,
      .data-section-surface,
      .data-empty {
        border: 1px solid rgba(18, 17, 15, 0.08);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.42);
      }

      .data-summary-card {
        display: grid;
        gap: 0.3rem;
        padding: 0.2rem 0 0.1rem;
        border: 0;
        background: transparent;
        border-radius: 0;
      }

      .data-summary-top {
        display: flex;
        align-items: baseline;
        gap: 0.48rem 0.6rem;
        flex-wrap: wrap;
      }

      .data-company-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        color: rgba(17, 17, 17, 0.9);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .data-company-tag::before {
        content: "";
        width: 0.42rem;
        height: 0.42rem;
        border-radius: 999px;
        background: currentColor;
        opacity: 0.9;
      }

      .data-company-tag-openai {
        color: var(--openai);
      }

      .data-company-tag-anthropic {
        color: var(--anthropic);
      }

      .data-kicker {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
      }

      .data-equation-line {
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 0.32rem 0.44rem;
        line-height: 1.1;
      }

      .data-equation-part {
        font-size: 0.9rem;
        font-weight: 680;
      }

      .data-equation-part-total {
        color: var(--ink);
        font-size: 1.24rem;
        font-weight: 780;
        letter-spacing: -0.02em;
      }

      .data-equation-part-floor {
        color: rgba(17, 17, 17, 0.82);
      }

      .data-equation-part-uplift {
        color: var(--muted);
      }

      .data-equation-symbol {
        color: var(--muted);
        font-size: 1rem;
        font-weight: 800;
        line-height: 1;
        padding-bottom: 0.34rem;
      }

      .data-point-meta {
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.35;
      }

      .data-method-note {
        margin-top: 0.08rem;
      }

      .data-method-note summary,
      .data-fold summary {
        cursor: pointer;
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        list-style: none;
      }

      .data-method-note summary::-webkit-details-marker,
      .data-fold summary::-webkit-details-marker {
        display: none;
      }

      .data-method-note[open] summary {
        margin-bottom: 0.35rem;
      }

      .data-method-note p {
        margin: 0;
        color: var(--ink);
        font-size: 0.73rem;
        line-height: 1.45;
      }

      .data-fold {
        margin-top: 0.04rem;
      }

      .data-fold[open] summary {
        margin-bottom: 0.32rem;
      }

      .data-section {
        display: grid;
        gap: 0.34rem;
        position: relative;
        padding-left: 0.9rem;
      }

      .data-section::before {
        content: "";
        position: absolute;
        left: 0.18rem;
        top: 0.15rem;
        bottom: 0.15rem;
        width: 1px;
        background: rgba(17, 17, 17, 0.09);
      }

      .data-section-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 0.75rem;
        flex-wrap: wrap;
      }

      .data-section-title {
        margin: 0;
        color: rgba(17, 17, 17, 0.74);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .data-section-copy {
        margin: 0.18rem 0 0;
        color: var(--muted);
        font-size: 0.71rem;
        line-height: 1.35;
      }

      .data-section-meta {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
      }

      .data-section-surface {
        padding: 0.18rem 0 0.12rem;
        border: 0;
        background: transparent;
        border-radius: 0;
      }

      .data-contributor-list,
      .data-future-list,
      .data-evidence-list {
        display: grid;
        gap: 0;
      }

      .data-item {
        border-top: 1px solid rgba(18, 17, 15, 0.045);
      }

      .data-item:first-child {
        border-top: 0;
      }

      .data-item summary {
        list-style: none;
        cursor: pointer;
        padding: 0.52rem 0;
      }

      .data-item summary::-webkit-details-marker {
        display: none;
      }

      .data-item > :not(summary) {
        display: none;
      }

      .data-item[open] > :not(summary) {
        display: block;
      }

      .data-item-main {
        display: grid;
        gap: 0.34rem;
      }

      .data-row-top,
      .data-item-top,
      .data-evidence-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.7rem;
      }

      .data-row-name,
      .data-item-name,
      .data-evidence-title {
        color: var(--ink);
        font-size: 0.77rem;
        font-weight: 760;
        line-height: 1.3;
      }

      .data-row-value,
      .data-item-value {
        color: rgba(17, 17, 17, 0.78);
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
      }

      .data-row-detail,
      .data-item-meta {
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1.35;
      }

      .data-item-progress,
      .data-row-bar {
        width: 100%;
        height: 2px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(18, 17, 15, 0.045);
      }

      .data-item-progress-fill,
      .data-row-bar-fill {
        height: 100%;
        border-radius: 999px;
      }

      .data-item-extra {
        padding: 0 0 0.72rem;
      }

      .data-row-note,
      .data-item-note,
      .data-evidence-copy {
        margin-top: 0.18rem;
        color: rgba(17, 17, 17, 0.76);
        font-size: 0.7rem;
        line-height: 1.42;
      }

      .data-rollup-foot {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding-top: 0.44rem;
        margin-top: 0.08rem;
        border-top: 1px solid rgba(18, 17, 15, 0.05);
      }

      .data-rollup-label {
        color: var(--muted);
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .data-rollup-value {
        color: rgba(17, 17, 17, 0.82);
        font-size: 0.72rem;
        font-weight: 700;
      }

      .data-fold-meta {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: none;
      }

      .data-link {
        display: inline-flex;
        align-items: flex-start;
        min-height: 1.7rem;
        padding: 0.22rem 0.52rem 0.18rem;
        border: 1px solid var(--data-link-border);
        border-radius: var(--control-radius);
        background: var(--data-link-bg);
        color: var(--data-link-text);
        text-decoration: none;
        font-weight: 700;
        font-size: 0.72rem;
        line-height: 1.2;
        white-space: normal;
      }

      .data-link:hover {
        background: var(--data-link-hover-bg);
        border-color: var(--data-link-hover-border);
      }

      .axis-label,
      .tick-label,
      .year-label,
      .section-title {
        fill: var(--muted);
        font-size: 14px;
        letter-spacing: 0.03em;
      }

      .axis-label {
        font-weight: 700;
      }

      .year-label {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.14em;
      }

      .section-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.18em;
      }

      .grid-line {
        stroke: var(--grid);
        stroke-width: 1;
        stroke-dasharray: 2 8;
      }

      .axis-line {
        stroke: var(--axis-line-color);
        stroke-width: 1.2;
      }

      .quarter-line {
        stroke: var(--quarter-line-color);
        stroke-width: 1;
      }

      .projection-region {
        fill: var(--projection-region-fill);
      }

      .projection-region-hatch {
        fill: url(#projection-hatch);
        opacity: 0.45;
      }

      .current-divider {
        stroke: var(--current-divider-color);
        stroke-width: 1.4;
        stroke-dasharray: 4 7;
      }

      .skeleton-line {
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        opacity: var(--skeleton-opacity);
      }

      .reveal-line,
      .reveal-glow,
      .projected-line,
      .projected-glow,
      .window-track,
      .window-path,
      .window-glow {
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
      }

      .head-halo {
        opacity: var(--head-halo-opacity);
      }

      .playhead-line {
        stroke: var(--playhead-line-color);
        stroke-width: 1.2;
        stroke-dasharray: 5 7;
      }

      .marker-release {
        stroke: var(--marker-stroke);
      }

      .event-label {
        pointer-events: none;
      }

      .event-label-box {
        fill: rgba(255, 255, 255, 0.94);
      }

      .event-label-fill {
        fill-opacity: 0.24;
      }

      .event-label-frame {
        fill: none;
        stroke-width: 1.2;
      }

      .event-label-title {
        font-size: 13.5px;
        font-weight: 800;
        letter-spacing: 0.01em;
        paint-order: stroke fill;
      }

      .label-text {
        font-weight: 800;
        letter-spacing: 0.01em;
        paint-order: stroke fill;
        font-size: 13px;
      }

      .label-inline {
        pointer-events: none;
      }

      .label-inline-connector {
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
      }

      .label-inline-box,
      .label-inline-fill {
        stroke-linejoin: round;
      }

      .label-inline-box {
        fill: var(--label-box-fill);
      }

      .label-inline-fill {
        fill-opacity: 0.88;
      }

      .playhead-chip text {
        font-size: 10px;
        font-weight: 760;
        letter-spacing: 0.12em;
        fill: var(--playhead-chip-text);
      }

      .hero-date-month,
      .hero-date-year {
        fill: var(--hero-date-fill);
        font-family: var(--hero-date-font);
        font-weight: 700;
        letter-spacing: -0.045em;
        pointer-events: none;
      }

      .hero-date-month {
        font-size: 52px;
      }

      .hero-date-year {
        font-size: 78px;
      }

      @media (max-width: 980px) {
        .story {
          gap: 0.45rem;
          padding: 0.75rem;
        }

        .poster {
          border-radius: 22px;
        }

        .masthead {
          gap: 0.45rem 0.8rem;
          padding: 0.8rem 0.9rem 0.2rem;
        }

        .tab-strip {
          padding: 0.16rem 0.9rem 0.28rem;
        }

        .stage-shell {
          padding: 0 0.3rem 0.45rem;
        }

        .data-shell {
          padding: 0 0.8rem 0.85rem;
        }
      }

      @media (max-width: 700px) {
        .story {
          gap: 0.35rem;
        }

        .masthead {
          gap: 0.35rem 0.6rem;
        }

        .title-block {
          gap: 0.14rem;
        }

        h1 {
          font-size: clamp(0.94rem, 3.1vw, 1.18rem);
          letter-spacing: -0.035em;
        }

        .subtitle {
          max-width: 32ch;
          font-size: 0.56rem;
          line-height: 1.18;
        }

        .legend {
          flex-direction: column;
          align-items: flex-end;
          gap: 0.22rem;
          margin-top: 0.12rem;
          font-size: 0.94rem;
          line-height: 1;
        }

        .legend-item {
          gap: 0.38rem;
        }

        .legend-line {
          width: 30px;
          border-top-width: 5px;
        }

        .playback-controls,
        .timeline-controls,
        .speed-controls,
        .theme-controls {
          gap: 0.3rem;
          font-size: 0.58rem;
        }

        .view-controls {
          gap: 0.35rem;
          font-size: 0.58rem;
        }

        .tab-strip {
          gap: 0.32rem;
          padding: 0.12rem 0.75rem 0.24rem;
          overflow-x: auto;
          scrollbar-width: none;
        }

        .tab-strip::-webkit-scrollbar {
          display: none;
        }

        .tab-button {
          min-width: 68px;
          padding: 0.34rem 0.62rem 0.3rem;
          font-size: 0.62rem;
        }

        .control-button,
        .view-toggle {
          min-width: 78px;
          padding: 0.34rem 0.58rem 0.31rem;
        }

        .control-button {
          min-width: 70px;
        }

        .timeline-range {
          width: 168px;
        }

        .speed-presets,
        .theme-presets {
          gap: 0.22rem;
        }

        .timeline-value {
          font-size: 0.54rem;
        }

        .speed-preset,
        .theme-preset {
          min-width: 46px;
          padding: 0.31rem 0.48rem 0.28rem;
        }

        .data-shell {
          gap: 0.58rem;
          padding: 0 0.72rem 0.72rem;
        }

        .data-toolbar {
          gap: 0.5rem;
        }

        .data-toolbar-right {
          gap: 0.42rem;
        }

        .data-pill,
        .data-download {
          font-size: 0.6rem;
          padding: 0.3rem 0.5rem 0.27rem;
        }

        .data-inspector-wrap {
          padding: 0.62rem 0.66rem 0.78rem;
        }

        .data-year-group {
          grid-template-columns: 1fr;
          gap: 0.5rem;
        }

        .data-year-label {
          font-size: 1.5rem;
          padding-top: 0;
        }

        .data-year-thread {
          padding-left: 0.82rem;
        }

        .data-equation-symbol {
          padding-bottom: 0.22rem;
        }

        .data-row-top,
        .data-item-top,
        .data-evidence-top,
        .data-section-header {
          gap: 0.45rem;
        }

        .story-controls {
          justify-content: flex-start;
          gap: 0.45rem;
          min-height: 2rem;
          padding: 0;
        }

        .hero-date-month {
          font-size: 28px;
        }

        .hero-date-year {
          font-size: 44px;
        }

        .event-label-title {
          font-size: 12px;
        }

        .label-text {
          font-size: 11.5px;
        }

      }
    </style>
  </head>
  <body>
    <main class="story">
      <section class="poster">
        <header class="masthead">
          <div class="title-block">
            <h1 id="title"></h1>
            <p class="subtitle" id="subtitle"></p>
          </div>
          <div class="legend" id="legend"></div>
        </header>
        <div class="tab-strip" role="tablist" aria-label="Page view">
          <button class="tab-button" id="tab-chart" type="button" role="tab" aria-selected="true">Chart</button>
          <button class="tab-button" id="tab-data" type="button" role="tab" aria-selected="false">Data</button>
        </div>

        <div class="panel-stack">
          <div class="stage-shell" id="chart-panel">
            <svg id="chart" role="img" aria-label="Animated chart of OpenAI and Anthropic compute buildout with flagship model training windows"></svg>
          </div>
          <section class="data-shell panel-hidden" id="data-panel" aria-label="Underlying chart data explorer">
            <div class="data-toolbar">
              <div class="data-company-filters" id="data-company-filters"></div>
              <div class="data-toolbar-right">
                <div class="data-meta" id="data-meta"></div>
                <button class="data-download" id="data-download" type="button">Download CSV</button>
              </div>
            </div>
            <div class="data-inspector-wrap">
              <div class="data-inspector" id="data-inspector"></div>
            </div>
          </section>
        </div>
      </section>
      <div class="story-controls" id="story-controls">
        <div class="playback-controls">
          <span class="control-label">Playback</span>
          <button class="control-button control-button-primary" id="play-toggle" type="button" aria-pressed="true">Pause</button>
          <button class="control-button" id="replay-button" type="button">Replay</button>
        </div>
        <div class="view-controls">
          <span class="control-label">Phase one</span>
          <button class="view-toggle" id="reveal-toggle" type="button" aria-pressed="true">Reveal on</button>
        </div>
        <div class="timeline-controls">
          <span class="control-label">Timeline</span>
          <input class="timeline-range" id="timeline-range" type="range" min="0" max="1000" step="1" value="0" aria-label="Timeline scrubber">
          <span class="timeline-value" id="timeline-value">Jan 2023</span>
        </div>
        <div class="speed-controls">
          <span class="speed-label control-label">Speed</span>
          <div class="speed-presets" id="speed-presets" aria-label="Animation speed presets"></div>
        </div>
      </div>
    </main>

    <script>
      const DATA = __PAYLOAD__;

      const DAY_MS = 24 * 60 * 60 * 1000;
      const DATE_FORMATTER = new Intl.DateTimeFormat("en-GB", {
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      });
      const MONTH_NAME_FORMATTER = new Intl.DateTimeFormat("en-GB", {
        month: "long",
        timeZone: "UTC",
      });
      const YEAR_ONLY_FORMATTER = new Intl.DateTimeFormat("en-GB", {
        year: "numeric",
        timeZone: "UTC",
      });

      const svg = document.getElementById("chart");
      const chartPanel = document.getElementById("chart-panel");
      const dataPanel = document.getElementById("data-panel");
      const stage = chartPanel;
      const storyControls = document.getElementById("story-controls");
      const tabChart = document.getElementById("tab-chart");
      const tabData = document.getElementById("tab-data");
      const playToggle = document.getElementById("play-toggle");
      const replayButton = document.getElementById("replay-button");
      const revealToggle = document.getElementById("reveal-toggle");
      const timelineRange = document.getElementById("timeline-range");
      const timelineValue = document.getElementById("timeline-value");
      const speedPresets = document.getElementById("speed-presets");
      const dataCompanyFilters = document.getElementById("data-company-filters");
      const dataMeta = document.getElementById("data-meta");
      const dataDownload = document.getElementById("data-download");
      const dataInspector = document.getElementById("data-inspector");

      document.getElementById("title").textContent = DATA.title;
      const subtitleEl = document.getElementById("subtitle");
      if (DATA.subtitle) {
        subtitleEl.textContent = DATA.subtitle;
      } else {
        subtitleEl.hidden = true;
      }
      const DATA_EXPLORER = DATA.data_explorer;

      const SPEED_PRESETS = [
        { step: -8, label: "0.25x" },
        { step: -4, label: "0.5x" },
        { step: 0, label: "1x" },
        { step: 4, label: "2x" },
        { step: 8, label: "4x" },
        { step: 12, label: "8x" },
      ];
      const BASE_THEME_VARS = {
        "--page": "#f5f2eb",
        "--surface": "rgba(255, 255, 255, 0.84)",
        "--surface-strong": "rgba(255, 255, 255, 0.96)",
        "--ink": "#12110f",
        "--muted": "#625b53",
        "--line": "rgba(18, 17, 15, 0.12)",
        "--grid": "rgba(18, 17, 15, 0.08)",
        "--body-font": "\\"Gill Sans\\", \\"Helvetica Neue\\", sans-serif",
        "--display-font": "\\"Helvetica Neue\\", \\"Arial Narrow\\", \\"Avenir Next Condensed\\", sans-serif",
        "--hero-date-font": "var(--display-font)",
        "--body-background": "linear-gradient(180deg, #f6f5f1 0%, #e7e4de 100%)",
        "--body-grid": "linear-gradient(rgba(18, 17, 15, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(18, 17, 15, 0.02) 1px, transparent 1px)",
        "--body-grid-size": "44px",
        "--body-grid-opacity": "1",
        "--body-grid-mask": "linear-gradient(to bottom, rgba(0, 0, 0, 0.55), transparent 80%)",
        "--poster-background": "linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 245, 239, 0.95))",
        "--poster-border": "rgba(18, 17, 15, 0.08)",
        "--poster-radius": "16px",
        "--poster-shadow": "0 10px 36px rgba(18, 17, 15, 0.08)",
        "--poster-blur": "none",
        "--poster-grid": "linear-gradient(rgba(18, 17, 15, 0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(18, 17, 15, 0.035) 1px, transparent 1px)",
        "--poster-grid-size": "120px",
        "--poster-grid-opacity": "1",
        "--poster-grid-mask": "linear-gradient(to right, rgba(0, 0, 0, 0.28), transparent 84%)",
        "--title-color": "#12110f",
        "--title-weight": "780",
        "--title-tracking": "-0.06em",
        "--title-transform": "none",
        "--legend-size": "1.16rem",
        "--legend-weight": "800",
        "--legend-line-width": "44px",
        "--legend-line-thickness": "5px",
        "--control-radius": "10px",
        "--control-bg": "rgba(255, 255, 255, 0.92)",
        "--control-border": "rgba(18, 17, 15, 0.16)",
        "--control-text": "rgba(18, 17, 15, 0.76)",
        "--control-active-bg": "#12110f",
        "--control-active-border": "#12110f",
        "--control-active-text": "rgba(255, 250, 245, 0.97)",
        "--control-shadow": "0 4px 12px rgba(18, 17, 15, 0.1)",
        "--control-transform": "uppercase",
        "--control-spacing": "0.08em",
        "--timeline-accent": "#12110f",
        "--table-radius": "12px",
        "--data-wrap-bg": "rgba(255, 255, 255, 0.82)",
        "--data-wrap-border": "rgba(18, 17, 15, 0.08)",
        "--data-head-bg": "rgba(247, 243, 235, 0.96)",
        "--data-row-alt": "rgba(18, 17, 15, 0.018)",
        "--data-link-bg": "rgba(49, 77, 99, 0.08)",
        "--data-link-border": "rgba(49, 77, 99, 0.22)",
        "--data-link-text": "#314d63",
        "--data-link-hover-bg": "rgba(49, 77, 99, 0.14)",
        "--data-link-hover-border": "rgba(49, 77, 99, 0.32)",
        "--axis-line-color": "rgba(18, 17, 15, 0.16)",
        "--quarter-line-color": "rgba(18, 17, 15, 0.05)",
        "--projection-region-fill": "rgba(18, 17, 15, 0.025)",
        "--projection-hatch-stroke": "rgba(219, 92, 43, 0.12)",
        "--current-divider-color": "rgba(18, 17, 15, 0.22)",
        "--playhead-line-color": "rgba(18, 17, 15, 0.18)",
        "--marker-stroke": "rgba(255, 255, 255, 0.96)",
        "--label-box-fill": "rgba(255, 255, 255, 0.96)",
        "--hero-date-fill": "rgba(18, 17, 15, 0.08)",
        "--playhead-chip-fill": "#12110f",
        "--playhead-chip-text": "#ffffff",
        "--head-halo-opacity": "0.16",
        "--reveal-glow-opacity": "0.1",
        "--skeleton-opacity": "0.1",
      };
      const BASE_THEME_COMPANY_COLORS = {
        openai: "#101010",
        anthropic: "#db5c2b",
      };
      const THEMES = [
        {
          key: "terminal",
          label: "Terminal",
          colorScheme: "light",
          glowMultiplier: 0.06,
          haloOpacity: 0.06,
          projectionOpacityMultiplier: 0.84,
          vars: {
            "--page": "#eceeea",
            "--surface": "rgba(252, 253, 249, 0.98)",
            "--surface-strong": "rgba(255, 255, 253, 0.99)",
            "--ink": "#111111",
            "--muted": "#4b4f54",
            "--line": "rgba(17, 17, 17, 0.08)",
            "--grid": "rgba(17, 17, 17, 0.08)",
            "--body-font": "\\"Menlo\\", \\"Monaco\\", \\"SFMono-Regular\\", monospace",
            "--display-font": "\\"Menlo\\", \\"Monaco\\", \\"SFMono-Regular\\", monospace",
            "--hero-date-font": "var(--display-font)",
            "--body-background": "linear-gradient(180deg, #f7f8f4 0%, #e3e6df 100%)",
            "--body-grid": "linear-gradient(rgba(17, 17, 17, 0.022) 1px, transparent 1px), linear-gradient(90deg, rgba(17, 17, 17, 0.022) 1px, transparent 1px)",
            "--body-grid-size": "18px",
            "--body-grid-mask": "linear-gradient(to bottom, rgba(0, 0, 0, 0.42), transparent 88%)",
            "--poster-background": "linear-gradient(180deg, rgba(255, 255, 252, 0.99), rgba(247, 249, 244, 0.98))",
            "--poster-border": "rgba(17, 17, 17, 0.14)",
            "--poster-radius": "6px",
            "--poster-shadow": "0 8px 20px rgba(17, 17, 17, 0.035)",
            "--poster-grid": "linear-gradient(rgba(17, 17, 17, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(17, 17, 17, 0.05) 1px, transparent 1px)",
            "--poster-grid-size": "20px",
            "--poster-grid-mask": "linear-gradient(to right, rgba(0, 0, 0, 0.38), transparent 94%)",
            "--title-color": "#111111",
            "--title-weight": "760",
            "--title-tracking": "-0.015em",
            "--title-transform": "uppercase",
            "--legend-size": "1.02rem",
            "--legend-line-width": "36px",
            "--legend-line-thickness": "4px",
            "--control-radius": "4px",
            "--control-bg": "rgba(255, 255, 253, 0.98)",
            "--control-border": "rgba(17, 17, 17, 0.14)",
            "--control-text": "#222222",
            "--control-active-bg": "#111111",
            "--control-active-border": "#111111",
            "--control-active-text": "#ffffff",
            "--control-shadow": "none",
            "--control-spacing": "0.08em",
            "--timeline-accent": "#111111",
            "--data-wrap-bg": "rgba(252, 253, 249, 0.94)",
            "--data-wrap-border": "rgba(17, 17, 17, 0.12)",
            "--data-head-bg": "rgba(237, 239, 234, 0.98)",
            "--data-row-alt": "rgba(17, 17, 17, 0.028)",
            "--data-link-bg": "rgba(17, 17, 17, 0.06)",
            "--data-link-border": "rgba(17, 17, 17, 0.16)",
            "--data-link-text": "#111111",
            "--data-link-hover-bg": "rgba(17, 17, 17, 0.1)",
            "--data-link-hover-border": "rgba(17, 17, 17, 0.24)",
            "--axis-line-color": "rgba(17, 17, 17, 0.16)",
            "--quarter-line-color": "rgba(17, 17, 17, 0.05)",
            "--projection-region-fill": "rgba(17, 17, 17, 0.025)",
            "--projection-hatch-stroke": "rgba(17, 17, 17, 0.08)",
            "--current-divider-color": "rgba(17, 17, 17, 0.2)",
            "--playhead-line-color": "rgba(17, 17, 17, 0.16)",
            "--marker-stroke": "rgba(255, 255, 253, 0.96)",
            "--label-box-fill": "rgba(255, 255, 253, 0.97)",
            "--hero-date-fill": "rgba(17, 17, 17, 0.08)",
            "--playhead-chip-fill": "#111111",
            "--playhead-chip-text": "#ffffff",
            "--head-halo-opacity": "0.06",
            "--reveal-glow-opacity": "0.02",
            "--skeleton-opacity": "0.09",
          },
          companyColors: {
            openai: "#111111",
            anthropic: "#e45722",
          },
        },
      ].map((theme) => ({
        ...theme,
        vars: { ...BASE_THEME_VARS, ...(theme.vars || {}) },
        companyColors: { ...BASE_THEME_COMPANY_COLORS, ...(theme.companyColors || {}) },
      }));
      const THEMES_BY_KEY = Object.fromEntries(THEMES.map((theme) => [theme.key, theme]));
      const legendItems = [
        { kind: "line", text: "OpenAI", color: "var(--openai)" },
        { kind: "line", text: "Anthropic", color: "var(--anthropic)" },
      ];
      const legendRoot = document.getElementById("legend");
      legendItems.forEach((item) => {
        const wrapper = document.createElement("span");
        wrapper.className = "legend-item";
        wrapper.style.color = item.color;
        const glyph = document.createElement("span");
        glyph.className =
          item.kind === "line"
            ? "legend-line"
            : item.kind === "window"
              ? "legend-window"
              : item.kind === "diamond"
                ? "legend-diamond"
                : "legend-circle";
        const text = document.createElement("span");
        text.textContent = item.text;
        wrapper.appendChild(glyph);
        wrapper.appendChild(text);
        legendRoot.appendChild(wrapper);
      });

      SPEED_PRESETS.forEach((preset) => {
        const button = document.createElement("button");
        button.className = "speed-preset";
        button.type = "button";
        button.dataset.speedStep = String(preset.step);
        button.textContent = preset.label;
        button.setAttribute("aria-pressed", preset.step === 0 ? "true" : "false");
        speedPresets?.appendChild(button);
      });

      const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      const makeSvg = (tag, attrs = {}) => {
        const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
      };

      const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
      const lerp = (start, end, progress) => start + (end - start) * progress;
      const easeOutCubic = (value) => 1 - Math.pow(1 - value, 3);
      const easeInOutCubic = (value) =>
        value < 0.5 ? 4 * value * value * value : 1 - Math.pow(-2 * value + 2, 3) / 2;
      const hexToRgba = (hex, alpha) => {
        const normalized = hex.replace("#", "");
        const safe = normalized.length === 3
          ? normalized.split("").map((value) => value + value).join("")
          : normalized;
        const numeric = Number.parseInt(safe, 16);
        const red = (numeric >> 16) & 255;
        const green = (numeric >> 8) & 255;
        const blue = numeric & 255;
        return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
      };

      const formatGw = (value) => {
        if (value >= 1) {
          return `${value.toFixed(1)} GW`;
        }
        return `${value.toFixed(2)} GW`;
      };

      const formatAxisGw = (value) => {
        if (value < 1) {
          return `${value.toFixed(value < 0.2 ? 2 : 1)} GW`;
        }
        if (Math.abs(value - Math.round(value)) < 0.001) {
          return `${Math.round(value)} GW`;
        }
        return `${value.toFixed(1)} GW`;
      };

      const axisCeiling = (value) => {
        if (value <= 2.2) {
          return 2.5;
        }
        if (value <= 3.2) {
          return 3.5;
        }
        if (value <= 6.2) {
          return Math.ceil(value);
        }
        return Math.ceil(value / 2) * 2;
      };

      const formatDate = (time) => DATE_FORMATTER.format(new Date(time));
      const formatHeroMonth = (time) => MONTH_NAME_FORMATTER.format(new Date(time)).toUpperCase();
      const formatHeroYear = (time) => YEAR_ONLY_FORMATTER.format(new Date(time));
      const formatDetailedGw = (value) => `${Number(value).toFixed(3).replace(/\\.?0+$/, "")} GW`;
      const formatRoundedMw = (valueGw) => `${Math.round(Number(valueGw) * 1000).toLocaleString()} MW`;
      const speedForStep = (step) => Math.pow(2, Number(step) / 4);
      const rootStyle = document.documentElement.style;

      const companies = DATA.companies.map((company) => {
        const points = company.points.map((point) => ({
          month_end: point.month_end,
          total_gw: point.total_gw,
          time: Date.parse(`${point.month_end}T00:00:00Z`),
        }));

        const models = company.models.map((model, index) => ({
          ...model,
          index,
          companyColor: company.color,
          trainingTime: Date.parse(`${model.training_start_date}T00:00:00Z`),
          releaseTime: Date.parse(`${model.release_date}T00:00:00Z`),
          fadeEndTime: Date.parse(`${model.release_date}T00:00:00Z`) + 96 * DAY_MS,
          labelFadeEndTime: Date.parse(`${model.release_date}T00:00:00Z`) + 220 * DAY_MS,
        }));

        return {
          ...company,
          baseColor: company.color,
          points,
          models,
          elements: null,
          activeLabel: null,
        };
      });
      const companyByName = Object.fromEntries(companies.map((company) => [company.name, company]));

      const colorForCompany = (companyName) => {
        const company = companyByName[companyName];
        return company?.currentColor || company?.baseColor || "#111111";
      };

      const storyStart = Date.parse(`${DATA.story_start_date}T00:00:00Z`);
      const storyEnd = Date.parse(`${DATA.story_end_date}T00:00:00Z`);
      const currentCutoff = Date.parse(`${DATA.current_cutoff_date}T00:00:00Z`);
      const PHASE_ONE_SHARE = 0.76;
      const PHASE_ONE_HOLD_MS = 950;
      const REVEAL_MIN_Y_MAX = 0.22;
      const REVEAL_MIN_SPAN_MS = 320 * DAY_MS;
      const REVEAL_LOOKAHEAD_MS = 110 * DAY_MS;

      const state = {
        activeTab: "chart",
        progress: prefersReducedMotion ? 1 : 0,
        playing: !prefersReducedMotion,
        dragging: false,
        resumeAfterDrag: false,
        resumeAfterTab: false,
        lastTimestamp: null,
        baseDurationMs: 18000,
        speedStep: 0,
        speedMultiplier: 1,
        themeKey: "terminal",
        revealMode: true,
        dataCompany: DATA_EXPLORER.default_company || "Both",
        phasePauseActive: false,
        phasePauseConsumed: prefersReducedMotion,
        phasePauseElapsedMs: 0,
        rafId: null,
      };

      let scene = null;

      function pointAtTime(points, time) {
        if (time <= points[0].time) {
          return { time, value: points[0].total_gw };
        }
        if (time >= points[points.length - 1].time) {
          return { time, value: points[points.length - 1].total_gw };
        }
        for (let index = 1; index < points.length; index += 1) {
          const current = points[index];
          if (time <= current.time) {
            const previous = points[index - 1];
            const span = current.time - previous.time;
            const progress = span === 0 ? 0 : (time - previous.time) / span;
            return {
              time,
              value: lerp(previous.total_gw, current.total_gw, progress),
            };
          }
        }
        return { time, value: points[points.length - 1].total_gw };
      }

      function sliceSeries(points, startTime, endTime) {
        const safeEnd = Math.max(startTime, endTime);
        const output = [];
        const startPoint = pointAtTime(points, startTime);
        output.push(startPoint);

        points.forEach((point) => {
          if (point.time > startTime && point.time < safeEnd) {
            output.push({ time: point.time, value: point.total_gw });
          }
        });

        if (safeEnd > startTime) {
          output.push(pointAtTime(points, safeEnd));
        }
        return output;
      }

      function pathFromPoints(points, geometry) {
        if (!points.length) {
          return "";
        }
        return points
          .map((point, index) => {
            const x = geometry.xForTime(point.time).toFixed(2);
            const y = geometry.yForValue(point.value).toFixed(2);
            return `${index === 0 ? "M" : "L"}${x},${y}`;
          })
          .join(" ");
      }

      function trimLeadingFlatPoints(points, epsilon = 0.0005) {
        if (points.length <= 1) {
          return points;
        }
        const initialValue = points[0].value;
        let firstChangeIndex = -1;
        for (let index = 1; index < points.length; index += 1) {
          if (Math.abs(points[index].value - initialValue) > epsilon) {
            firstChangeIndex = index;
            break;
          }
        }
        if (firstChangeIndex === -1) {
          return points.slice(-1);
        }
        return points.slice(firstChangeIndex);
      }

      function buildLabelMetrics(parentGroup) {
        const group = makeSvg("g", {
          opacity: 0,
          "pointer-events": "none",
        });
        const title = makeSvg("text", {
          class: "event-label-title",
          fill: "transparent",
          "text-anchor": "middle",
          "dominant-baseline": "middle",
        });
        group.appendChild(title);
        parentGroup.appendChild(group);
        return {
          group,
          title,
          width: 0,
          height: 0,
          textWidth: 0,
          key: "",
          layout: null,
        };
      }

      function buildInlineLabel(parentGroup, defsRoot, clipId) {
        const group = makeSvg("g", { class: "label-inline", opacity: 0 });
        const connector = makeSvg("line", { class: "label-inline-connector" });
        const clipPath = makeSvg("clipPath", { id: clipId });
        const clipRect = makeSvg("rect");
        clipPath.appendChild(clipRect);
        defsRoot.appendChild(clipPath);
        const contentGroup = makeSvg("g", { "clip-path": `url(#${clipId})` });
        const box = makeSvg("rect", { class: "label-inline-box" });
        const fill = makeSvg("rect", { class: "label-inline-fill" });
        const text = makeSvg("text", {
          class: "label-text",
          "text-anchor": "middle",
          "dominant-baseline": "middle",
        });
        group.appendChild(connector);
        contentGroup.appendChild(box);
        contentGroup.appendChild(fill);
        contentGroup.appendChild(text);
        group.appendChild(contentGroup);
        parentGroup.appendChild(group);
        return { group, connector, contentGroup, clipRect, box, fill, text };
      }

      function labelTitleForModel(model) {
        return model.story_label || model.label_name;
      }

      function refreshLabelMetrics(label, model) {
        const nextKey = model.model_key;
        const nextHeight = scene.isCompact ? 28 : 32;
        if (label.key === nextKey && label.height === nextHeight) {
          return;
        }
        label.key = nextKey;
        label.title.textContent = labelTitleForModel(model);
        label.title.setAttribute("x", 0);
        label.title.setAttribute("y", 0);
        const titleBounds = label.title.getBBox();
        const horizontalPadding = scene.isCompact ? 18 : 26;
        label.textWidth = Math.ceil(titleBounds.width);
        label.width = Math.max(scene.isCompact ? 72 : 84, label.textWidth + horizontalPadding);
        label.height = nextHeight;
      }

      function labelLifecycle(model, playheadTime) {
        const visibleStart = Math.max(model.trainingTime, storyStart);
        if (playheadTime < visibleStart) {
          return null;
        }
        const releaseAge = playheadTime - model.releaseTime;
        const releaseFlash = scene.isCompact ? 26 * DAY_MS : 34 * DAY_MS;
        const releaseHold = scene.isCompact ? 94 * DAY_MS : 126 * DAY_MS;
        const memoryFade = scene.isCompact ? 180 * DAY_MS : 240 * DAY_MS;
        const memoryOpacity = scene.isCompact ? 0.22 : 0.32;

        if (playheadTime < model.releaseTime) {
          return {
            phase: "training",
            opacity: 1,
            releaseEmphasis: 0,
          };
        }

        if (releaseAge <= releaseFlash) {
          return {
            phase: "released-flash",
            opacity: 1,
            releaseEmphasis: 1 - clamp(releaseAge / releaseFlash, 0, 1),
          };
        }

        if (releaseAge <= releaseHold) {
          return {
            phase: "released",
            opacity: 1,
            releaseEmphasis: 0,
          };
        }

        if (playheadTime <= currentCutoff) {
          const settleProgress = clamp((releaseAge - releaseHold) / memoryFade, 0, 1);
          return {
            phase: "memory",
            opacity: lerp(1, memoryOpacity, settleProgress),
            releaseEmphasis: 0,
          };
        }

        const projectionFade = clamp((playheadTime - currentCutoff) / (120 * DAY_MS), 0, 1);
        return {
          phase: "memory",
          opacity: lerp(memoryOpacity, 0, projectionFade),
          releaseEmphasis: 0,
        };
      }

      function markerPulse(ageDays) {
        if (ageDays < 0 || ageDays > 24) {
          return 0;
        }
        return 1 - clamp(ageDays / 24, 0, 1);
      }

      function maxValueThrough(time) {
        let maxValue = 0;
        companies.forEach((company) => {
          company.points.forEach((point) => {
            if (point.time <= time) {
              maxValue = Math.max(maxValue, point.total_gw);
            }
          });
          maxValue = Math.max(maxValue, pointAtTime(company.points, time).value);
        });
        return maxValue;
      }

      function phaseOneRevealYMax(playheadTime) {
        const visibleMax = maxValueThrough(playheadTime);
        const phaseProgress = clamp((playheadTime - storyStart) / (currentCutoff - storyStart), 0, 1);
        const headroomFactor = lerp(1.62, 1.18, easeInOutCubic(phaseProgress));
        const headroomBias = lerp(0.04, 0.12, phaseProgress);
        return clamp(
          visibleMax * headroomFactor + headroomBias,
          REVEAL_MIN_Y_MAX,
          scene.currentCeilingMax,
        );
      }

      function phaseOneRevealDomainEnd(playheadTime) {
        const phaseProgress = clamp((playheadTime - storyStart) / (currentCutoff - storyStart), 0, 1);
        const minSpan = Math.min(REVEAL_MIN_SPAN_MS, currentCutoff - storyStart);
        const growthEnd = lerp(
          storyStart + minSpan,
          currentCutoff,
          easeOutCubic(phaseProgress),
        );
        const lookaheadEnd = Math.min(
          currentCutoff,
          playheadTime + lerp(REVEAL_LOOKAHEAD_MS, 170 * DAY_MS, phaseProgress),
        );
        return clamp(
          Math.max(growthEnd, lookaheadEnd),
          storyStart + minSpan,
          currentCutoff,
        );
      }

      function currentPlayheadTime() {
        if (state.progress <= PHASE_ONE_SHARE) {
          return lerp(storyStart, currentCutoff, state.progress / PHASE_ONE_SHARE);
        }
        const projectionProgress = easeInOutCubic((state.progress - PHASE_ONE_SHARE) / (1 - PHASE_ONE_SHARE));
        return lerp(currentCutoff, storyEnd, projectionProgress);
      }

      function frameGeometryFor(playheadTime) {
        const projectionProgressRaw = clamp((state.progress - PHASE_ONE_SHARE) / (1 - PHASE_ONE_SHARE), 0, 1);
        const projectionProgress = easeInOutCubic(projectionProgressRaw);
        const revealActive = state.revealMode && projectionProgressRaw === 0 && playheadTime < currentCutoff;
        const phaseOneDomainEnd = revealActive
          ? phaseOneRevealDomainEnd(playheadTime)
          : currentCutoff;
        const domainEndTime = projectionProgressRaw === 0
          ? phaseOneDomainEnd
          : lerp(currentCutoff, storyEnd, projectionProgress);
        const yMax = projectionProgressRaw === 0
          ? revealActive
            ? phaseOneRevealYMax(playheadTime)
            : scene.currentCeilingMax
          : lerp(scene.currentCeilingMax, scene.futureCeilingMax, projectionProgress);
        const xForTime = (time) => scene.plotLeft + ((time - storyStart) / (domainEndTime - storyStart)) * scene.plotWidth;
        const yForValue = (value) => scene.plotBottom - (value / yMax) * scene.plotHeight;
        return {
          playheadTime,
          projectionProgress,
          revealActive,
          domainEndTime,
          yMax,
          xForTime,
          yForValue,
        };
      }

      function assignLabelRails(company) {
        const railEndXs = [];
        [...company.models]
          .sort((left, right) => left.trainingTime - right.trainingTime)
          .forEach((model) => {
            const label = model.elements.label;
            const visibleStartX = scene.phaseOneXForTime(Math.max(model.trainingTime, storyStart));
            const visibleEndX = scene.phaseOneXForTime(model.releaseTime);
            const intervalWidth = Math.max(8, visibleEndX - visibleStartX);
            const visualWidth = Math.max(label.width, intervalWidth);
            const visualStartX = Math.min(visibleStartX, scene.plotRight - visualWidth);
            const railGapX = scene.isCompact ? 4 : 10;
            let railIndex = railEndXs.findIndex((endX) => visualStartX >= endX + railGapX);
            if (railIndex === -1) {
              railIndex = railEndXs.length;
              railEndXs.push(visualStartX + visualWidth);
            } else {
              railEndXs[railIndex] = visualStartX + visualWidth;
            }
            model.labelRail = railIndex;
          });
        company.labelRailCount = railEndXs.length;
      }

      function buildIntervalLabelLayout(company, model, geometry = null) {
        const label = model.elements.label;
        const xForTime = geometry?.xForTime ?? scene.phaseOneXForTime;
        const yForValue = geometry?.yForValue ?? scene.phaseOneYForValue;
        const visibleStart = Math.max(model.trainingTime, storyStart);
        const intervalStartX = clamp(xForTime(visibleStart), scene.plotLeft, scene.plotRight);
        const intervalEndX = clamp(xForTime(model.releaseTime), scene.plotLeft, scene.plotRight);
        const intervalWidth = Math.max(8, intervalEndX - intervalStartX);
        const width = Math.max(label.width, intervalWidth);
        const x = clamp(intervalStartX, scene.plotLeft, scene.plotRight - width);
        const y = company.key === "openai"
          ? (() => {
              const railGap = scene.isCompact ? 27 : 31;
              const topInset = scene.isCompact ? 64 : 100;
              const shelfY = scene.plotTop + topInset + model.labelRail * railGap;
              const anchorTime = lerp(visibleStart, model.releaseTime, 0.7);
              const anchorPoint = pointAtTime(company.points, anchorTime);
              const anchorY = yForValue(anchorPoint.value);
              const curveTargetY = anchorY - label.height - (scene.isCompact ? 10 : 13) - model.labelRail * (scene.isCompact ? 7 : 9);
              const pullStrength = easeInOutCubic(clamp((anchorPoint.value - 0.6) / 1.7, 0, 1)) * 0.42;
              return clamp(
                lerp(shelfY, curveTargetY, pullStrength),
                scene.plotTop + 18,
                scene.plotBottom - label.height - 18,
              );
            })()
          : scene.plotBottom - label.height - (scene.isCompact ? 6 : 14) - model.labelRail * (scene.isCompact ? 26 : 31);
        return {
          x,
          y,
          width,
          intervalStartX,
          intervalEndX,
          fillMaxWidth: clamp(intervalEndX - x, 0, width),
        };
      }

      function refreshFixedLabelLayouts() {
        companies.forEach((company) => {
          company.models.forEach((model) => {
            refreshLabelMetrics(model.elements.label, model);
          });
          assignLabelRails(company);
          company.models.forEach((model) => {
            model.elements.label.layout = buildIntervalLabelLayout(company, model);
          });
        });
      }

      function hideLabelGroup(labelObject) {
        labelObject?.group?.setAttribute("opacity", 0);
      }

      function describeLabelState(company, model, lifecycle, playheadTime) {
        const label = model.elements.label;
        const liveLayout = scene.currentFrame?.revealActive
          ? buildIntervalLabelLayout(company, model, scene.currentFrame)
          : label.layout;
        if (!lifecycle || lifecycle.opacity <= 0.01 || !liveLayout) {
          return null;
        }
        const visibleStart = Math.max(model.trainingTime, storyStart);
        const visibleDuration = Math.max(DAY_MS, model.releaseTime - visibleStart);
        const trainingProgress = clamp((playheadTime - visibleStart) / visibleDuration, 0, 1);
        const fillProgress = lifecycle.phase === "training" ? trainingProgress : 1;
        const releasedNow = lifecycle.phase === "released-flash" || lifecycle.phase === "released";
        const memoryState = lifecycle.phase === "memory";
        const midpointTime = lerp(visibleStart, model.releaseTime, 0.56);
        const midpointValue = pointAtTime(company.points, midpointTime).value;
        const liveTime = releasedNow || memoryState
          ? model.releaseTime
          : lerp(visibleStart, model.releaseTime, Math.max(0.02, fillProgress));
        const liveValue = pointAtTime(company.points, liveTime).value;
        return {
          label,
          layout: liveLayout,
          trainingProgress,
          fillProgress,
          releasedNow,
          memoryState,
          opacity: lifecycle.opacity,
          releaseFlashStrength: lifecycle.releaseEmphasis ?? 0,
          intervalSpan: Math.max(8, liveLayout.intervalEndX - liveLayout.intervalStartX),
          midpoint: {
            x: scene.currentFrame.xForTime(midpointTime),
            y: scene.currentFrame.yForValue(midpointValue),
          },
          livePoint: {
            x: scene.currentFrame.xForTime(liveTime),
            y: scene.currentFrame.yForValue(liveValue),
          },
          releasePoint: {
            x: scene.currentFrame.xForTime(model.releaseTime),
            y: scene.currentFrame.yForValue(model.release_gw),
          },
        };
      }

      function inlineBaseGeometry(company, model, descriptor) {
        const { label, midpoint } = descriptor;
        const width = clamp(label.textWidth + (scene.isCompact ? 16 : 22), scene.isCompact ? 70 : 82, scene.isCompact ? 148 : 176);
        const height = scene.isCompact ? 18 : 22;
        const railStep = scene.isCompact ? 18 : 22;
        const baseOffset = company.key === "openai" ? -(scene.isCompact ? 22 : 26) : (scene.isCompact ? 16 : 20);
        const y = clamp(
          midpoint.y + baseOffset + (company.key === "openai" ? -model.labelRail * railStep : model.labelRail * railStep),
          scene.plotTop + 10,
          scene.plotBottom - height - 10,
        );
        const x = clamp(midpoint.x - width / 2, scene.plotLeft, scene.plotRight - width);
        return {
          x,
          y,
          width,
          height,
          targetX: clamp(midpoint.x, x + 10, x + width - 10),
          targetY: company.key === "openai" ? y + height : y,
        };
      }

      function drawInlineShape(inline, company, model, descriptor, geometry, options = {}) {
        const { fillProgress, releasedNow, memoryState, opacity } = descriptor;
        const {
          x,
          y,
          width,
          height,
          targetX,
          targetY,
          anchorX,
          anchorY,
        } = geometry;
        const textOpacity = options.textOpacity ?? 1;
        const connectorOpacity = (options.connectorOpacity ?? 1) * opacity;
        const clipProgress = clamp(options.clipProgress ?? 1, 0, 1);
        const effectiveFillProgress = options.fillProgressOverride ?? fillProgress;

        inline.group.setAttribute("opacity", opacity.toFixed(3));
        inline.connector.setAttribute("x1", anchorX.toFixed(2));
        inline.connector.setAttribute("y1", anchorY.toFixed(2));
        inline.connector.setAttribute("x2", targetX.toFixed(2));
        inline.connector.setAttribute("y2", targetY.toFixed(2));
        inline.connector.setAttribute("stroke", hexToRgba(model.companyColor, memoryState ? 0.34 : 0.56));
        inline.connector.setAttribute("stroke-width", (scene.isCompact ? 1.5 : 1.7).toFixed(2));
        inline.connector.setAttribute("opacity", connectorOpacity.toFixed(3));

        inline.box.setAttribute("x", x.toFixed(2));
        inline.box.setAttribute("y", y.toFixed(2));
        inline.box.setAttribute("width", width.toFixed(2));
        inline.box.setAttribute("height", height.toFixed(2));
        inline.box.setAttribute("rx", (height / 2).toFixed(2));
        inline.box.setAttribute("stroke", model.companyColor);
        inline.box.setAttribute("stroke-width", releasedNow || memoryState ? "1.2" : "1.35");
        inline.box.style.fill = releasedNow ? model.companyColor : memoryState ? hexToRgba(model.companyColor, 0.8) : "rgba(255,255,255,0.94)";

        inline.fill.setAttribute("x", x.toFixed(2));
        inline.fill.setAttribute("y", y.toFixed(2));
        inline.fill.setAttribute("width", (width * effectiveFillProgress).toFixed(2));
        inline.fill.setAttribute("height", height.toFixed(2));
        inline.fill.setAttribute("rx", (height / 2).toFixed(2));
        inline.fill.setAttribute("fill", model.companyColor);
        inline.fill.setAttribute("opacity", releasedNow || memoryState ? "0" : "0.26");

        inline.text.textContent = labelTitleForModel(model);
        inline.text.setAttribute("x", (x + width / 2).toFixed(2));
        inline.text.setAttribute("y", (y + height / 2 + 0.5).toFixed(2));
        inline.text.style.fill = releasedNow || memoryState ? "#ffffff" : model.companyColor;
        inline.text.style.stroke = releasedNow || memoryState ? hexToRgba(model.companyColor, 0.32) : "rgba(255,255,255,0.95)";
        inline.text.style.strokeWidth = releasedNow || memoryState ? "1.2px" : "2.8px";
        inline.text.style.opacity = String(textOpacity);

        inline.clipRect.setAttribute("x", x.toFixed(2));
        inline.clipRect.setAttribute("y", y.toFixed(2));
        inline.clipRect.setAttribute("width", (width * clipProgress).toFixed(2));
        inline.clipRect.setAttribute("height", height.toFixed(2));
        inline.clipRect.setAttribute("rx", (height / 2).toFixed(2));
      }

      function updateInlineLabel(company, model, lifecycle, playheadTime) {
        const inline = model.elements.inlineLabel;
        const descriptor = describeLabelState(company, model, lifecycle, playheadTime);
        if (!descriptor) {
          hideLabelGroup(inline);
          return;
        }

        const finalGeometry = inlineBaseGeometry(company, model, descriptor);
        const livePoint = descriptor.livePoint;
        const threshold = 0.22;
        if (descriptor.fillProgress < threshold && !descriptor.releasedNow && !descriptor.memoryState) {
          const stubSize = scene.isCompact ? 12 : 14;
          drawInlineShape(
            inline,
            company,
            model,
            descriptor,
            {
              x: livePoint.x - stubSize / 2,
              y: livePoint.y - stubSize / 2,
              width: stubSize,
              height: stubSize,
              targetX: livePoint.x,
              targetY: livePoint.y,
              anchorX: livePoint.x,
              anchorY: livePoint.y,
            },
            {
              textOpacity: 0,
              connectorOpacity: 0,
              fillProgressOverride: 1,
            },
          );
          inline.box.style.fill = model.companyColor;
          inline.box.setAttribute("stroke", "#ffffff");
          inline.box.setAttribute("stroke-width", "1.4");
          inline.fill.setAttribute("opacity", "0");
          inline.clipRect.setAttribute("width", stubSize.toFixed(2));
          return;
        }
        const revealT = descriptor.releasedNow || descriptor.memoryState
          ? 1
          : easeOutCubic(clamp((descriptor.fillProgress - threshold) / 0.16, 0, 1));
        const stubSize = scene.isCompact ? 12 : 14;
        drawInlineShape(
          inline,
          company,
          model,
          descriptor,
          {
            x: lerp(livePoint.x - stubSize / 2, finalGeometry.x, revealT),
            y: lerp(livePoint.y - stubSize / 2, finalGeometry.y, revealT),
            width: lerp(stubSize, finalGeometry.width, revealT),
            height: lerp(stubSize, finalGeometry.height, revealT),
            targetX: lerp(livePoint.x, finalGeometry.targetX, revealT),
            targetY: lerp(livePoint.y, finalGeometry.targetY, revealT),
            anchorX: livePoint.x,
            anchorY: livePoint.y,
          },
          {
            textOpacity: descriptor.releasedNow || descriptor.memoryState ? 1 : revealT,
            connectorOpacity: descriptor.releasedNow || descriptor.memoryState ? 1 : revealT,
          },
        );
      }

      function updateModelLabel(company, model, lifecycle, playheadTime) {
        updateInlineLabel(company, model, lifecycle, playheadTime);
      }

      function buildScene() {
        const bounds = stage.getBoundingClientRect();
        const width = Math.max(360, Math.round(bounds.width));
        const height = Math.max(520, Math.round(bounds.height));
        const isCompact = width <= 820;
        const theme = currentTheme();
        const revealGlowOpacity = clamp(
          Number.parseFloat(theme.vars["--reveal-glow-opacity"] || "0.1") * (theme.glowMultiplier ?? 1),
          0,
          1,
        );
        const margin = isCompact
          ? { top: 46, right: 18, bottom: 52, left: 90 }
          : { top: 44, right: 24, bottom: 58, left: 108 };
        const plotLeft = margin.left;
        const plotRight = width - margin.right;
        const plotTop = margin.top;
        const plotBottom = height - margin.bottom;
        const plotWidth = plotRight - plotLeft;
        const plotHeight = plotBottom - plotTop;
        const phaseOneXForTime = (time) =>
          plotLeft + ((time - storyStart) / (currentCutoff - storyStart)) * plotWidth;
        const currentMaxValue = Math.max(
          ...companies.flatMap((company) =>
            company.points
              .filter((point) => point.time <= currentCutoff)
              .map((point) => point.total_gw),
          ),
        );
        const futureMaxValue = Math.max(...companies.flatMap((company) => company.points.map((point) => point.total_gw)));
        const currentCeilingMax = axisCeiling(currentMaxValue);
        const futureCeilingMax = axisCeiling(futureMaxValue);
        const phaseOneYForValue = (value) => plotBottom - (value / currentCeilingMax) * plotHeight;

        svg.textContent = "";
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

        const defs = makeSvg("defs");
        const projectionPattern = makeSvg("pattern", {
          id: "projection-hatch",
          patternUnits: "userSpaceOnUse",
          width: 12,
          height: 12,
          patternTransform: "rotate(32)",
        });
        projectionPattern.appendChild(
          makeSvg("line", {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 12,
            stroke: "var(--projection-hatch-stroke)",
            "stroke-width": 2,
          }),
        );
        defs.appendChild(projectionPattern);
        svg.appendChild(defs);

        const staticGroup = makeSvg("g");
        const dynamicGroup = makeSvg("g");
        const labelLayer = makeSvg("g");
        const labelMetricsLayer = makeSvg("g");
        const inlineLabelLayer = makeSvg("g");
        labelLayer.appendChild(labelMetricsLayer);
        labelLayer.appendChild(inlineLabelLayer);
        svg.appendChild(staticGroup);
        svg.appendChild(dynamicGroup);
        svg.appendChild(labelLayer);

        const projectionWash = makeSvg("rect", {
          class: "projection-region",
          x: plotRight,
          y: plotTop,
          width: 0,
          height: plotHeight,
          opacity: 0,
        });
        const projectionHatch = makeSvg("rect", {
          class: "projection-region-hatch",
          x: plotRight,
          y: plotTop,
          width: 0,
          height: plotHeight,
          opacity: 0,
        });
        const currentDivider = makeSvg("line", {
          class: "current-divider",
          x1: plotRight,
          x2: plotRight,
          y1: plotTop,
          y2: plotBottom,
          opacity: 0,
        });
        staticGroup.appendChild(projectionWash);
        staticGroup.appendChild(projectionHatch);
        staticGroup.appendChild(currentDivider);

        const currentSectionTitle = makeSvg("text", {
          class: "section-title",
          "text-anchor": "middle",
          opacity: 0,
        });
        currentSectionTitle.textContent = "CURRENT";
        const projectedSectionTitle = makeSvg("text", {
          class: "section-title",
          "text-anchor": "middle",
          opacity: 0,
        });
        projectedSectionTitle.textContent = isCompact ? "FUTURE" : "PROJECTED";
        const heroDateMonth = makeSvg("text", {
          class: "hero-date-month",
          "text-anchor": "start",
        });
        const heroDateYear = makeSvg("text", {
          class: "hero-date-year",
          "text-anchor": "start",
        });
        staticGroup.appendChild(heroDateMonth);
        staticGroup.appendChild(heroDateYear);
        staticGroup.appendChild(currentSectionTitle);
        staticGroup.appendChild(projectedSectionTitle);

        const yAxisElements = Array.from({ length: 6 }, (_, index) => {
          const line = makeSvg("line", { class: index === 0 ? "axis-line" : "grid-line" });
          const label = makeSvg("text", {
            "text-anchor": "end",
            class: "tick-label",
          });
          staticGroup.appendChild(line);
          staticGroup.appendChild(label);
          return { line, label };
        });

        const yAxisTitle = makeSvg("text", {
          class: "axis-label",
          "text-anchor": "middle",
        });
        yAxisTitle.textContent = "Compute (GW)";
        staticGroup.appendChild(yAxisTitle);

        const yearElements = [];
        for (let year = 2023; year <= new Date(storyEnd).getUTCFullYear(); year += 1) {
          const time = Date.parse(`${year}-01-31T00:00:00Z`);
          const line = makeSvg("line", { class: "axis-line" });
          const label = makeSvg("text", {
            "text-anchor": "middle",
            class: "year-label",
          });
          label.textContent = year.toString();
          staticGroup.appendChild(line);
          staticGroup.appendChild(label);
          yearElements.push({ time, line, label });
        }

        const quarterElements = [];
        for (let year = 2023; year <= new Date(storyEnd).getUTCFullYear(); year += 1) {
          ["04-30", "07-31", "10-31"].forEach((monthEnd) => {
            const time = Date.parse(`${year}-${monthEnd}T00:00:00Z`);
            const line = makeSvg("line", { class: "quarter-line" });
            staticGroup.appendChild(line);
            quarterElements.push({ time, line });
          });
        }

        const playheadLine = makeSvg("line", {
          class: "playhead-line",
          x1: plotLeft,
          x2: plotLeft,
          y1: plotTop,
          y2: plotBottom,
        });
        dynamicGroup.appendChild(playheadLine);

        const playheadChip = makeSvg("g", { class: "playhead-chip" });
        const playheadChipRect = makeSvg("rect", {
          x: plotLeft - 34,
          y: plotBottom + 14,
          width: 68,
          height: 22,
          rx: 11,
          fill: "var(--playhead-chip-fill)",
        });
        const playheadChipText = makeSvg("text", {
          x: plotLeft,
          y: plotBottom + 29,
          "text-anchor": "middle",
        });
        playheadChip.appendChild(playheadChipRect);
        playheadChip.appendChild(playheadChipText);
        dynamicGroup.appendChild(playheadChip);

        companies.forEach((company) => {
          const companyGroup = makeSvg("g");
          const windowsGroup = makeSvg("g");
          const markersGroup = makeSvg("g");
          const lineGroup = makeSvg("g");

          const skeletonPath = makeSvg("path", {
            d: "",
            class: "skeleton-line",
            stroke: company.color,
            "stroke-width": isCompact ? 2.4 : 2.8,
          });
          const revealGlow = makeSvg("path", {
            class: "reveal-glow",
            stroke: company.color,
            "stroke-width": isCompact ? 9 : 11,
            opacity: revealGlowOpacity.toFixed(3),
          });
          const revealLine = makeSvg("path", {
            class: "reveal-line",
            stroke: company.color,
            "stroke-width": isCompact ? 3.6 : 4.2,
          });
          const projectedGlow = makeSvg("path", {
            class: "projected-glow",
            stroke: company.color,
            "stroke-width": isCompact ? 7.6 : 9.4,
            "stroke-dasharray": isCompact ? "5 7" : "7 8",
            opacity: 0,
          });
          const projectedLine = makeSvg("path", {
            class: "projected-line",
            stroke: company.color,
            "stroke-width": isCompact ? 3.1 : 3.6,
            "stroke-dasharray": isCompact ? "5 7" : "7 8",
            opacity: 0,
          });
          const headHalo = makeSvg("circle", {
            class: "head-halo",
            r: isCompact ? 8 : 10,
            fill: company.color,
          });
          const headDot = makeSvg("circle", {
            r: isCompact ? 4.2 : 4.8,
            fill: company.color,
            stroke: "#fff",
            "stroke-width": 1.4,
          });

          lineGroup.appendChild(skeletonPath);
          lineGroup.appendChild(revealGlow);
          lineGroup.appendChild(revealLine);
          lineGroup.appendChild(projectedGlow);
          lineGroup.appendChild(projectedLine);
          lineGroup.appendChild(headHalo);
          lineGroup.appendChild(headDot);

          company.models.forEach((model) => {
            const track = makeSvg("path", {
              class: "window-track",
              stroke: "rgba(255,255,255,0.9)",
              "stroke-width": isCompact ? 6.4 : 7.4,
              opacity: 0,
            });
            const glow = makeSvg("path", {
              class: "window-glow",
              stroke: company.color,
              "stroke-width": isCompact ? 8 : 10,
              opacity: 0,
            });
            const line = makeSvg("path", {
              class: "window-path",
              stroke: company.color,
              "stroke-width": isCompact ? 4 : 4.5,
              opacity: 0,
            });
            windowsGroup.appendChild(track);
            windowsGroup.appendChild(glow);
            windowsGroup.appendChild(line);

            const releaseMarker = makeSvg("circle", {
              class: "marker-release",
              r: isCompact ? 4.1 : 4.5,
              fill: company.color,
              "stroke-width": 1.6,
              opacity: 0,
            });

            markersGroup.appendChild(releaseMarker);

            const label = buildLabelMetrics(labelMetricsLayer);
            const inlineLabel = buildInlineLabel(
              inlineLabelLayer,
              defs,
              `inline-clip-${company.key}-${model.index}`,
            );
            model.elements = {
              track,
              glow,
              line,
              releaseMarker,
              label,
              inlineLabel,
            };
          });
          company.elements = {
            group: companyGroup,
            windowsGroup,
            markersGroup,
            lineGroup,
            skeletonPath,
            revealGlow,
            revealLine,
            projectedGlow,
            projectedLine,
            headHalo,
            headDot,
          };

          companyGroup.appendChild(windowsGroup);
          companyGroup.appendChild(markersGroup);
          companyGroup.appendChild(lineGroup);
          dynamicGroup.appendChild(companyGroup);
        });

        scene = {
          width,
          height,
          isCompact,
          margin,
          plotLeft,
          plotRight,
          plotTop,
          plotBottom,
          plotWidth,
          plotHeight,
          phaseOneXForTime,
          phaseOneYForValue,
          currentCeilingMax,
          futureCeilingMax,
          projectionWash,
          projectionHatch,
          currentDivider,
          currentSectionTitle,
          projectedSectionTitle,
          heroDateMonth,
          heroDateYear,
          playheadLine,
          playheadChipRect,
          playheadChipText,
          labelLayer,
          inlineLabelLayer,
          staticGroup,
          yAxisElements,
          yAxisTitle,
          yearElements,
          quarterElements,
        };
        refreshFixedLabelLayouts();
      }

      function updateAxes(frame) {
        scene.yAxisElements.forEach((item, index) => {
          const ratio = index / (scene.yAxisElements.length - 1);
          const tickValue = frame.yMax * ratio;
          const y = frame.yForValue(tickValue);
          item.line.setAttribute("x1", scene.plotLeft);
          item.line.setAttribute("x2", scene.plotRight);
          item.line.setAttribute("y1", y.toFixed(2));
          item.line.setAttribute("y2", y.toFixed(2));
          item.label.setAttribute("x", (scene.plotLeft - 12).toFixed(2));
          item.label.setAttribute("y", (y + 4).toFixed(2));
          item.label.textContent = formatAxisGw(tickValue);
        });

        const axisTitleX = scene.isCompact ? 26 : 30;
        const axisTitleY = scene.plotTop + scene.plotHeight / 2;
        scene.yAxisTitle.setAttribute("x", axisTitleX.toFixed(2));
        scene.yAxisTitle.setAttribute("y", axisTitleY.toFixed(2));
        scene.yAxisTitle.setAttribute(
          "transform",
          `rotate(-90 ${axisTitleX.toFixed(2)} ${axisTitleY.toFixed(2)})`,
        );

        scene.yearElements.forEach((item) => {
          const visible = item.time >= storyStart && item.time <= frame.domainEndTime + DAY_MS;
          if (!visible) {
            item.line.setAttribute("opacity", 0);
            item.label.setAttribute("opacity", 0);
            return;
          }
          const x = frame.xForTime(item.time);
          item.line.setAttribute("x1", x.toFixed(2));
          item.line.setAttribute("x2", x.toFixed(2));
          item.line.setAttribute("y1", scene.plotTop);
          item.line.setAttribute("y2", scene.plotBottom);
          item.line.setAttribute("opacity", 1);
          item.label.setAttribute("x", x.toFixed(2));
          item.label.setAttribute("y", (scene.plotBottom + 24).toFixed(2));
          item.label.setAttribute("opacity", 1);
        });

        scene.quarterElements.forEach((item) => {
          const visible = item.time >= storyStart && item.time <= frame.domainEndTime + DAY_MS;
          if (!visible) {
            item.line.setAttribute("opacity", 0);
            return;
          }
          const x = frame.xForTime(item.time);
          item.line.setAttribute("x1", x.toFixed(2));
          item.line.setAttribute("x2", x.toFixed(2));
          item.line.setAttribute("y1", scene.plotTop);
          item.line.setAttribute("y2", scene.plotBottom);
          item.line.setAttribute("opacity", 1);
        });
      }

      function renderFrame() {
        if (!scene) {
          return;
        }

        const theme = currentTheme();
        const glowMultiplier = theme.glowMultiplier ?? 1;
        const haloOpacity = clamp(
          theme.haloOpacity ?? Number.parseFloat(theme.vars["--head-halo-opacity"] || "0.16"),
          0,
          1,
        );
        const revealGlowOpacity = clamp(
          Number.parseFloat(theme.vars["--reveal-glow-opacity"] || "0.1") * glowMultiplier,
          0,
          1,
        );
        const playheadTime = currentPlayheadTime();
        const frame = frameGeometryFor(playheadTime);
        scene.xForTime = frame.xForTime;
        scene.yForValue = frame.yForValue;
        scene.domainEndTime = frame.domainEndTime;
        scene.currentFrame = frame;
        updateAxes(frame);
        const playheadX = frame.xForTime(playheadTime);
        const cutoffX = frame.xForTime(currentCutoff);
        const projectionActive = state.progress >= PHASE_ONE_SHARE - 0.0001;
        const projectionVisible = frame.domainEndTime > currentCutoff + DAY_MS;
        const projectionWidth = projectionVisible
          ? Math.max(0, scene.plotRight - cutoffX)
          : 0;
        const projectionOpacity = projectionVisible
          ? clamp(
              lerp(0.2, 0.72, frame.projectionProgress) * (theme.projectionOpacityMultiplier ?? 1),
              0,
              1,
            )
          : 0;
        scene.projectionWash.setAttribute("x", cutoffX.toFixed(2));
        scene.projectionWash.setAttribute("y", scene.plotTop.toFixed(2));
        scene.projectionWash.setAttribute("width", projectionWidth.toFixed(2));
        scene.projectionWash.setAttribute("height", scene.plotHeight.toFixed(2));
        scene.projectionWash.setAttribute("opacity", (projectionOpacity * 0.9).toFixed(3));
        scene.projectionHatch.setAttribute("x", cutoffX.toFixed(2));
        scene.projectionHatch.setAttribute("y", scene.plotTop.toFixed(2));
        scene.projectionHatch.setAttribute("width", projectionWidth.toFixed(2));
        scene.projectionHatch.setAttribute("height", scene.plotHeight.toFixed(2));
        scene.projectionHatch.setAttribute("opacity", (projectionOpacity * 0.55).toFixed(3));
        scene.currentDivider.setAttribute("x1", cutoffX.toFixed(2));
        scene.currentDivider.setAttribute("x2", cutoffX.toFixed(2));
        scene.currentDivider.setAttribute("y1", scene.plotTop.toFixed(2));
        scene.currentDivider.setAttribute("y2", scene.plotBottom.toFixed(2));
        scene.currentDivider.setAttribute("opacity", projectionActive ? Math.max(0.62, projectionOpacity).toFixed(3) : 0);
        const sectionTitleY = scene.isCompact ? scene.plotTop - 12 : scene.plotTop - 14;
        const currentSectionX = clamp(
          scene.plotLeft + Math.max(44, (cutoffX - scene.plotLeft) * 0.5),
          scene.plotLeft + 34,
          cutoffX - 32,
        );
        const projectedSectionX = clamp(
          cutoffX + Math.max(42, (scene.plotRight - cutoffX) * 0.5),
          cutoffX + 34,
          scene.plotRight - 34,
        );
        const heroDateX = scene.plotLeft + (scene.isCompact ? 18 : 28);
        const heroMonthY = scene.plotTop + (scene.isCompact ? 38 : 56);
        const heroYearY = scene.plotTop + (scene.isCompact ? 78 : 122);
        scene.heroDateMonth.setAttribute("x", heroDateX.toFixed(2));
        scene.heroDateMonth.setAttribute("y", heroMonthY.toFixed(2));
        scene.heroDateMonth.textContent = formatHeroMonth(playheadTime);
        scene.heroDateYear.setAttribute("x", heroDateX.toFixed(2));
        scene.heroDateYear.setAttribute("y", heroYearY.toFixed(2));
        scene.heroDateYear.textContent = formatHeroYear(playheadTime);
        scene.currentSectionTitle.setAttribute("x", currentSectionX.toFixed(2));
        scene.currentSectionTitle.setAttribute("y", sectionTitleY.toFixed(2));
        scene.currentSectionTitle.setAttribute("opacity", projectionActive ? Math.max(0.7, projectionOpacity).toFixed(3) : 0);
        scene.projectedSectionTitle.setAttribute("x", projectedSectionX.toFixed(2));
        scene.projectedSectionTitle.setAttribute("y", sectionTitleY.toFixed(2));
        scene.projectedSectionTitle.setAttribute("opacity", projectionActive ? Math.max(0.7, projectionOpacity).toFixed(3) : 0);
        scene.playheadLine.setAttribute("x1", playheadX.toFixed(2));
        scene.playheadLine.setAttribute("x2", playheadX.toFixed(2));
        scene.playheadChipRect.setAttribute("x", (playheadX - 34).toFixed(2));
        scene.playheadChipText.setAttribute("x", playheadX.toFixed(2));
        scene.playheadChipText.textContent = formatDate(playheadTime).toUpperCase();
        syncTimelineControls(playheadTime);

        companies.forEach((company) => {
          const currentPoint = pointAtTime(company.points, playheadTime);
          const companyDisplayStart = company.models.length
            ? Math.max(
                storyStart,
                Math.min(...company.models.map((model) => model.trainingTime)),
              )
            : storyStart;
          const skeletonEndTime = frame.revealActive ? playheadTime : frame.domainEndTime;
          const skeletonPoints = trimLeadingFlatPoints(sliceSeries(company.points, companyDisplayStart, skeletonEndTime));
          const skeletonPath = pathFromPoints(skeletonPoints, frame);
          const historicalRevealEnd = Math.min(playheadTime, currentCutoff);
          const revealPoints = trimLeadingFlatPoints(sliceSeries(company.points, companyDisplayStart, historicalRevealEnd));
          const revealPath = pathFromPoints(revealPoints, frame);
          const projectedVisible = playheadTime > currentCutoff + DAY_MS;
          const projectedPoints = projectedVisible
            ? sliceSeries(company.points, currentCutoff, playheadTime)
            : [];
          const projectedPath = projectedPoints.length > 1 ? pathFromPoints(projectedPoints, frame) : "";
          const headX = frame.xForTime(playheadTime);
          const headY = frame.yForValue(currentPoint.value);

          company.elements.skeletonPath.setAttribute("d", skeletonPath);
          company.elements.revealGlow.setAttribute("d", revealPath);
          company.elements.revealLine.setAttribute("d", revealPath);
          company.elements.projectedGlow.setAttribute("d", projectedPath);
          company.elements.projectedLine.setAttribute("d", projectedPath);
          company.elements.revealGlow.setAttribute("opacity", revealPoints.length > 1 ? revealGlowOpacity.toFixed(3) : 0);
          company.elements.revealLine.setAttribute("opacity", revealPoints.length > 1 ? 1 : 0);
          company.elements.projectedGlow.setAttribute(
            "opacity",
            projectedVisible
              ? clamp((0.18 + frame.projectionProgress * 0.12) * glowMultiplier, 0, 1).toFixed(3)
              : 0,
          );
          company.elements.projectedLine.setAttribute("opacity", projectedVisible ? (0.72 + frame.projectionProgress * 0.16).toFixed(3) : 0);
          company.elements.headHalo.setAttribute("cx", headX.toFixed(2));
          company.elements.headHalo.setAttribute("cy", headY.toFixed(2));
          company.elements.headHalo.setAttribute("opacity", haloOpacity.toFixed(3));
          company.elements.headDot.setAttribute("cx", headX.toFixed(2));
          company.elements.headDot.setAttribute("cy", headY.toFixed(2));

          company.models.forEach((model) => {
            const releaseAgeDays = (playheadTime - model.releaseTime) / DAY_MS;

            const releasePoint = {
              x: frame.xForTime(model.releaseTime),
              y: frame.yForValue(model.release_gw),
            };

            const trainVisible = playheadTime >= model.trainingTime;
            const releaseVisible = playheadTime >= model.releaseTime;
            const segmentEnd = Math.min(playheadTime, model.releaseTime);
            if (trainVisible && segmentEnd > model.trainingTime) {
              const segmentPoints = trimLeadingFlatPoints(
                sliceSeries(company.points, model.trainingTime, segmentEnd),
              );
              const segmentPath = pathFromPoints(segmentPoints, frame);
              const active = playheadTime < model.releaseTime;
              const settledRelease = clamp(Math.max(0, releaseAgeDays) / 160, 0, 1);
              const lineOpacity = active ? 0.62 : lerp(0.42, 0.24, settledRelease);
              const glowOpacity = clamp((active ? 0.22 : lineOpacity * 0.34) * glowMultiplier, 0, 1);
              const trackOpacity = active ? 0.72 : lerp(0.58, 0.42, settledRelease);
              model.elements.track.setAttribute("d", segmentPath);
              model.elements.glow.setAttribute("d", segmentPath);
              model.elements.line.setAttribute("d", segmentPath);
              const segmentVisible = segmentPoints.length > 1;
              model.elements.track.setAttribute("opacity", segmentVisible ? trackOpacity.toFixed(3) : 0);
              model.elements.glow.setAttribute("opacity", segmentVisible ? glowOpacity.toFixed(3) : 0);
              model.elements.line.setAttribute("opacity", segmentVisible ? lineOpacity.toFixed(3) : 0);
            } else {
              model.elements.track.setAttribute("opacity", 0);
              model.elements.glow.setAttribute("opacity", 0);
              model.elements.line.setAttribute("opacity", 0);
            }

            const releasePulse = markerPulse(releaseAgeDays);
            const releaseRadius = (scene.isCompact ? 4.4 : 4.9) + releasePulse * 3.1;
            const releaseOpacity = releaseVisible
              ? Math.max(0.52, 1 - Math.min(0.26, releaseAgeDays / 220))
              : 0;
            model.elements.releaseMarker.setAttribute("cx", releasePoint.x.toFixed(2));
            model.elements.releaseMarker.setAttribute("cy", releasePoint.y.toFixed(2));
            model.elements.releaseMarker.setAttribute("r", releaseRadius.toFixed(2));
            model.elements.releaseMarker.setAttribute("opacity", releaseOpacity.toFixed(3));
            updateModelLabel(company, model, labelLifecycle(model, playheadTime), playheadTime);
          });
        });
      }

      function tick(timestamp) {
        if (!state.playing || state.dragging) {
          return;
        }
        if (state.lastTimestamp === null) {
          state.lastTimestamp = timestamp;
        }
        const delta = timestamp - state.lastTimestamp;
        state.lastTimestamp = timestamp;

        if (state.phasePauseActive) {
          state.phasePauseElapsedMs += delta;
          renderFrame();
          if (state.phasePauseElapsedMs >= PHASE_ONE_HOLD_MS) {
            state.phasePauseActive = false;
            state.phasePauseElapsedMs = 0;
            state.lastTimestamp = timestamp;
          }
          state.rafId = requestAnimationFrame(tick);
          return;
        }

        const nextProgress = Math.min(
          1,
          state.progress + (delta * state.speedMultiplier) / state.baseDurationMs,
        );

        if (
          !state.phasePauseConsumed &&
          state.progress < PHASE_ONE_SHARE &&
          nextProgress >= PHASE_ONE_SHARE
        ) {
          state.progress = PHASE_ONE_SHARE;
          state.phasePauseConsumed = true;
          state.phasePauseActive = true;
          state.phasePauseElapsedMs = 0;
          renderFrame();
          state.rafId = requestAnimationFrame(tick);
          return;
        }

        state.progress = nextProgress;
        renderFrame();
        if (state.progress >= 1) {
          state.playing = false;
          state.lastTimestamp = null;
          state.rafId = null;
          syncPlaybackControls();
          return;
        }
        state.rafId = requestAnimationFrame(tick);
      }

      function ensureAnimation() {
        if (!state.playing || state.dragging || state.rafId !== null) {
          return;
        }
        state.rafId = requestAnimationFrame(tick);
      }

      function play() {
        state.playing = true;
        syncPlaybackControls();
        ensureAnimation();
      }

      function pause() {
        state.playing = false;
        state.lastTimestamp = null;
        if (state.rafId !== null) {
          cancelAnimationFrame(state.rafId);
          state.rafId = null;
        }
        syncPlaybackControls();
      }

      function replay() {
        state.progress = 0;
        state.phasePauseActive = false;
        state.phasePauseConsumed = false;
        state.phasePauseElapsedMs = 0;
        state.lastTimestamp = null;
        renderFrame();
        play();
        syncPlaybackControls();
      }

      function updateSpeed(nextStep) {
        const minStep = SPEED_PRESETS[0].step;
        const maxStep = SPEED_PRESETS[SPEED_PRESETS.length - 1].step;
        const boundedStep = clamp(Number(nextStep), minStep, maxStep);
        const speed = speedForStep(boundedStep);
        state.speedStep = boundedStep;
        state.speedMultiplier = speed;
        syncSpeedControls();
        return speed;
      }

      function syncSpeedControls() {
        speedPresets?.querySelectorAll(".speed-preset").forEach((button) => {
          const pressed = Number(button.dataset.speedStep) === state.speedStep;
          button.setAttribute("aria-pressed", pressed ? "true" : "false");
        });
      }

      function currentTheme() {
        return THEMES_BY_KEY[state.themeKey] || THEMES[0];
      }

      function applyTheme(nextThemeKey, options = {}) {
        const theme = THEMES_BY_KEY[nextThemeKey] || THEMES[0];
        state.themeKey = theme.key;
        document.body.dataset.theme = theme.key;
        document.documentElement.style.colorScheme = theme.colorScheme || "light";

        Object.entries(theme.vars).forEach(([key, value]) => {
          rootStyle.setProperty(key, value);
        });
        Object.entries(theme.companyColors).forEach(([key, value]) => {
          rootStyle.setProperty(`--${key}`, value);
        });

        companies.forEach((company) => {
          const nextColor = theme.companyColors[company.key] || company.baseColor;
          company.color = nextColor;
          company.models.forEach((model) => {
            model.companyColor = nextColor;
          });
        });

        if (options.rebuild === false) {
          return theme.key;
        }

        if (state.activeTab === "chart") {
          rebuild();
        } else {
          renderDataExplorer();
        }

        return theme.key;
      }

      function syncPlaybackControls() {
        if (!playToggle) {
          return;
        }
        playToggle.setAttribute("aria-pressed", state.playing ? "true" : "false");
        playToggle.textContent = state.playing ? "Pause" : "Play";
        playToggle.classList.toggle("control-button-primary", state.playing);
      }

      function syncTimelineControls(playheadTime = currentPlayheadTime()) {
        if (timelineRange) {
          timelineRange.value = String(Math.round(state.progress * 1000));
        }
        if (timelineValue) {
          timelineValue.textContent = formatDate(playheadTime);
        }
      }

      function syncTabControls() {
        const chartActive = state.activeTab === "chart";
        tabChart?.setAttribute("aria-selected", chartActive ? "true" : "false");
        tabData?.setAttribute("aria-selected", chartActive ? "false" : "true");
        chartPanel?.classList.toggle("panel-hidden", !chartActive);
        dataPanel?.classList.toggle("panel-hidden", chartActive);
        storyControls?.classList.toggle("panel-hidden", !chartActive);
      }

      function currentDownloadDataset() {
        return DATA_EXPLORER.download;
      }

      function flattenCellValue(value) {
        if (Array.isArray(value)) {
          return value
            .map((item) => {
              if (item && typeof item === "object") {
                const label = String(item.label ?? "").trim();
                const href = String(item.href ?? "").trim();
                if (label && href) {
                  return `${label} (${href})`;
                }
                return label || href;
              }
              return String(item ?? "").trim();
            })
            .filter(Boolean)
            .join(" | ");
        }
        if (value && typeof value === "object") {
          return Object.values(value).map((item) => String(item ?? "").trim()).filter(Boolean).join(" | ");
        }
        return String(value ?? "");
      }

      function escapeCsvCell(value) {
        const normalized = flattenCellValue(value).replace(/\\r?\\n+/g, " ").trim();
        if (/[",\\n]/.test(normalized)) {
          return `"${normalized.replaceAll('"', '""')}"`;
        }
        return normalized;
      }

      function datasetCsvText(dataset) {
        const header = dataset.columns.join(",");
        const rows = dataset.rows.map((row) =>
          dataset.columns.map((column) => escapeCsvCell(row[column])).join(","),
        );
        return [header, ...rows].join("\\n");
      }

      function downloadCurrentDataset() {
        const dataset = currentDownloadDataset();
        if (!dataset) {
          return;
        }
        const csvText = datasetCsvText(dataset);
        const blob = new Blob([csvText], { type: "text/csv;charset=utf-8" });
        const href = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = href;
        link.download = dataset.filename || "anthropic-openai-compute-breakdown.csv";
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(href), 1000);
      }

      function dataPointsForCompany(company = state.dataCompany) {
        if (company === "Both") {
          return [...DATA_EXPLORER.points];
        }
        return DATA_EXPLORER.points.filter((point) => point.company === company);
      }

      function renderDataCompanyFilters() {
        if (!dataCompanyFilters) {
          return;
        }
        dataCompanyFilters.textContent = "";
        const presentCompanies = new Set(DATA_EXPLORER.points.map((point) => point.company));
        const options = ["Both", "OpenAI", "Anthropic"].filter((option) => option === "Both" || presentCompanies.has(option));
        if (!options.includes(state.dataCompany)) {
          state.dataCompany = DATA_EXPLORER.default_company || "Both";
        }
        options.forEach((option) => {
          const button = document.createElement("button");
          button.className = "data-pill";
          button.type = "button";
          button.dataset.companyValue = option;
          button.textContent = option;
          button.setAttribute("aria-pressed", state.dataCompany === option ? "true" : "false");
          dataCompanyFilters.appendChild(button);
        });
      }

      function createLinkStack(links) {
        if (!links?.length) {
          return null;
        }
        const stack = document.createElement("div");
        stack.className = "data-link-stack";
        links.forEach((item) => {
          const href = item?.href;
          if (!href) {
            return;
          }
          const link = document.createElement("a");
          link.className = "data-link";
          link.href = href;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = item?.label || "Open source";
          stack.appendChild(link);
        });
        return stack.childNodes.length ? stack : null;
      }

      function createSectionHeader(title, copy, metaText) {
        const header = document.createElement("div");
        header.className = "data-section-header";

        const left = document.createElement("div");
        const heading = document.createElement("h2");
        heading.className = "data-section-title";
        heading.textContent = title;
        left.appendChild(heading);
        if (copy) {
          const paragraph = document.createElement("p");
          paragraph.className = "data-section-copy";
          paragraph.textContent = copy;
          left.appendChild(paragraph);
        }
        header.appendChild(left);

        if (metaText) {
          const meta = document.createElement("div");
          meta.className = "data-section-meta";
          meta.textContent = metaText;
          header.appendChild(meta);
        }
        return header;
      }

      function pluralize(count, singular, plural = `${singular}s`) {
        return `${count} ${count === 1 ? singular : plural}`;
      }

      function companySortIndex(company) {
        return ["OpenAI", "Anthropic"].indexOf(company);
      }

      function yearLabelForPoint(point) {
        return String(point.date).slice(0, 4);
      }

      function periodLabelForPoint(point) {
        const year = yearLabelForPoint(point);
        return point.period.replace(year, "").replace(/\\s+/g, " ").trim() || "Year-end";
      }

      function createSummaryCard(point) {
        const summary = document.createElement("section");
        summary.className = "data-summary-card";

        const top = document.createElement("div");
        top.className = "data-summary-top";
        const companyTag = document.createElement("div");
        companyTag.className = `data-company-tag data-company-tag-${point.company.toLowerCase()}`;
        companyTag.textContent = point.company;
        top.appendChild(companyTag);

        const kicker = document.createElement("div");
        kicker.className = "data-kicker";
        kicker.textContent = periodLabelForPoint(point);
        top.appendChild(kicker);
        summary.appendChild(top);

        const equation = document.createElement("div");
        equation.className = "data-equation-line";
        const total = document.createElement("span");
        total.className = "data-equation-part data-equation-part-total";
        total.textContent = `${formatDetailedGw(point.total_gw)} total`;
        equation.appendChild(total);
        const equals = document.createElement("span");
        equals.className = "data-equation-symbol";
        equals.textContent = "=";
        equation.appendChild(equals);
        const floor = document.createElement("span");
        floor.className = "data-equation-part data-equation-part-floor";
        floor.textContent = `${formatDetailedGw(point.floor_gw)} named sites`;
        equation.appendChild(floor);
        const plus = document.createElement("span");
        plus.className = "data-equation-symbol";
        plus.textContent = "+";
        equation.appendChild(plus);
        const uplift = document.createElement("span");
        uplift.className = "data-equation-part data-equation-part-uplift";
        uplift.textContent = `${formatDetailedGw(point.uplift_gw)} other compute`;
        equation.appendChild(uplift);
        summary.appendChild(equation);

        const note = document.createElement("p");
        note.className = "data-point-meta";
        note.textContent =
          (point.range_low_gw !== null && point.range_high_gw !== null
            ? `${formatDetailedGw(point.range_low_gw)} to ${formatDetailedGw(point.range_high_gw)} range`
            : point.point_choice) +
          ` · ${pluralize(point.component_count, "named site")}`;
        summary.appendChild(note);

        const method = document.createElement("details");
        method.className = "data-method-note";
        const methodSummary = document.createElement("summary");
        methodSummary.textContent = ["Basis", point.estimate_basis, point.assumption_ids].filter(Boolean).join(" · ");
        method.appendChild(methodSummary);
        const methodText = document.createElement("p");
        methodText.textContent = point.summary_note;
        method.appendChild(methodText);
        summary.appendChild(method);

        return summary;
      }

      function createContributorItem(component, maxFloorGw, companyColor, totalFloorGw) {
        const item = document.createElement("details");
        item.className = "data-item";

        const summary = document.createElement("summary");
        const main = document.createElement("div");
        main.className = "data-item-main";

        const top = document.createElement("div");
        top.className = "data-item-top";
        const name = document.createElement("div");
        name.className = "data-item-name";
        name.textContent = component.data_center;
        top.appendChild(name);
        const value = document.createElement("div");
        value.className = "data-item-value";
        const floorShare = totalFloorGw > 0 ? (component.allocated_power_gw / totalFloorGw) * 100 : 0;
        value.textContent = `${formatDetailedGw(component.allocated_power_gw)} · ${floorShare.toFixed(0)}% of sites`;
        top.appendChild(value);
        main.appendChild(top);

        const meta = document.createElement("div");
        meta.className = "data-item-meta";
        const detailBits = [
          component.project,
          component.country,
          `${component.operational_buildings} building${component.operational_buildings === 1 ? "" : "s"}`,
          `${component.allocated_h100_equivalents.toLocaleString()} H100e`,
          pluralize(component.sources.length, "source"),
        ].filter(Boolean);
        meta.textContent = detailBits.join(" | ");
        main.appendChild(meta);

        const progress = document.createElement("div");
        progress.className = "data-item-progress";
        const fill = document.createElement("div");
        fill.className = "data-item-progress-fill";
        fill.style.width = `${Math.max(6, (component.allocated_power_gw / Math.max(maxFloorGw, 0.001)) * 100)}%`;
        fill.style.background = hexToRgba(companyColor, 0.68);
        progress.appendChild(fill);
        main.appendChild(progress);
        summary.appendChild(main);
        item.appendChild(summary);

        const extra = document.createElement("div");
        extra.className = "data-item-extra";
        const note = document.createElement("div");
        note.className = "data-item-note";
        note.textContent = component.timeline_status;
        extra.appendChild(note);
        const links = createLinkStack(component.sources);
        if (links) {
          links.style.marginTop = "0.45rem";
          extra.appendChild(links);
        }
        item.appendChild(extra);

        return item;
      }

      function createFutureSiteItem(site, companyColor) {
        const item = document.createElement("details");
        item.className = "data-item";

        const summary = document.createElement("summary");
        const main = document.createElement("div");
        main.className = "data-item-main";

        const top = document.createElement("div");
        top.className = "data-item-top";
        const name = document.createElement("div");
        name.className = "data-item-name";
        name.textContent = site.data_center;
        top.appendChild(name);
        const value = document.createElement("div");
        value.className = "data-item-value";
        value.textContent = `0 now → ${formatDetailedGw(site.first_live_power_gw)}`;
        top.appendChild(value);
        main.appendChild(top);

        const meta = document.createElement("div");
        meta.className = "data-item-meta";
        meta.textContent = [site.project, site.country, `First live ${site.first_live_date}`, pluralize(site.sources.length, "source")].filter(Boolean).join(" | ");
        main.appendChild(meta);

        const progress = document.createElement("div");
        progress.className = "data-item-progress";
        const fill = document.createElement("div");
        fill.className = "data-item-progress-fill";
        fill.style.width = `${Math.max(8, Math.min(100, site.first_live_power_gw * 100))}%`;
        fill.style.background = hexToRgba(companyColor, 0.52);
        progress.appendChild(fill);
        main.appendChild(progress);
        summary.appendChild(main);
        item.appendChild(summary);

        const extra = document.createElement("div");
        extra.className = "data-item-extra";
        const note = document.createElement("div");
        note.className = "data-item-note";
        note.textContent = site.note;
        extra.appendChild(note);
        const links = createLinkStack(site.sources);
        if (links) {
          links.style.marginTop = "0.45rem";
          extra.appendChild(links);
        }
        item.appendChild(extra);

        return item;
      }

      function createEvidenceItem(item) {
        const entry = document.createElement("details");
        entry.className = "data-item";

        const summary = document.createElement("summary");
        const main = document.createElement("div");
        main.className = "data-item-main";

        const top = document.createElement("div");
        top.className = "data-evidence-top";
        const title = document.createElement("div");
        title.className = "data-evidence-title";
        title.textContent = item.title;
        top.appendChild(title);
        main.appendChild(top);

        const meta = document.createElement("div");
        meta.className = "data-item-meta";
        meta.textContent = [item.role, item.evidence_type].filter(Boolean).join(" | ");
        main.appendChild(meta);
        summary.appendChild(main);
        entry.appendChild(summary);

        const extra = document.createElement("div");
        extra.className = "data-item-extra";
        if (item.quant_signal) {
          const quant = document.createElement("div");
          quant.className = "data-evidence-copy";
          quant.textContent = item.quant_signal;
          extra.appendChild(quant);
        }
        if (item.note) {
          const note = document.createElement("div");
          note.className = "data-evidence-copy";
          note.textContent = item.note;
          extra.appendChild(note);
        }
        const links = createLinkStack([{ href: item.href, label: "Source" }]);
        if (links) {
          links.style.marginTop = "0.45rem";
          extra.appendChild(links);
        }
        entry.appendChild(extra);

        return entry;
      }

      function createEmptyCard(message) {
        const empty = document.createElement("div");
        empty.className = "data-empty";
        empty.textContent = message;
        return empty;
      }

      function createPointFlow(point) {
        const companyColor = colorForCompany(point.company);
        const pointFlow = document.createElement("section");
        pointFlow.className = "data-point-flow";
        pointFlow.appendChild(createSummaryCard(point));

        const floorSection = document.createElement("section");
        floorSection.className = "data-section";
        floorSection.appendChild(
          createSectionHeader(
            "Named sites",
            "",
            `${pluralize(point.component_count, "site")} · ${formatDetailedGw(point.floor_gw)}`,
          ),
        );
        const floorSurface = document.createElement("div");
        floorSurface.className = "data-section-surface";
        if (point.floor_components.length) {
          const list = document.createElement("div");
          list.className = "data-contributor-list";
          const maxFloor = Math.max(...point.floor_components.map((component) => component.allocated_power_gw));
          point.floor_components.forEach((component) => {
            list.appendChild(createContributorItem(component, maxFloor, companyColor, point.floor_gw));
          });
          floorSurface.appendChild(list);
          const rollupFoot = document.createElement("div");
          rollupFoot.className = "data-rollup-foot";
          const label = document.createElement("div");
          label.className = "data-rollup-label";
          label.textContent = "Named sites subtotal";
          rollupFoot.appendChild(label);
          const value = document.createElement("div");
          value.className = "data-rollup-value";
          value.textContent = `${formatDetailedGw(point.floor_gw)} of ${formatDetailedGw(point.total_gw)} total`;
          rollupFoot.appendChild(value);
          floorSurface.appendChild(rollupFoot);
        } else {
          floorSurface.appendChild(createEmptyCard("No named live sites at this point."));
        }
        floorSection.appendChild(floorSurface);
        pointFlow.appendChild(floorSection);

        const evidenceFold = document.createElement("details");
        evidenceFold.className = "data-fold";
        const evidenceSummary = document.createElement("summary");
        evidenceSummary.textContent = "Other compute evidence";
        const evidenceMeta = document.createElement("span");
        evidenceMeta.className = "data-fold-meta";
        evidenceMeta.textContent = ` · ${pluralize(point.uplift_evidence.length, "source")} · ${formatDetailedGw(point.uplift_gw)}`;
        evidenceSummary.appendChild(evidenceMeta);
        evidenceFold.appendChild(evidenceSummary);

        const evidenceSection = document.createElement("section");
        evidenceSection.className = "data-section";
        const evidenceSurface = document.createElement("div");
        evidenceSurface.className = "data-section-surface";
        if (point.uplift_evidence.length) {
          const list = document.createElement("div");
          list.className = "data-evidence-list";
          point.uplift_evidence.forEach((item) => {
            list.appendChild(createEvidenceItem(item));
          });
          evidenceSurface.appendChild(list);
        } else {
          evidenceSurface.appendChild(createEmptyCard("No extra evidence for this point."));
        }
        evidenceSection.appendChild(evidenceSurface);
        evidenceFold.appendChild(evidenceSection);
        pointFlow.appendChild(evidenceFold);

        if (point.future_sites.length) {
          const futureDetails = document.createElement("details");
          futureDetails.className = "data-fold";
          const futureSummary = document.createElement("summary");
          futureSummary.textContent = `Not live yet`;
          const futureMeta = document.createElement("span");
          futureMeta.className = "data-fold-meta";
          futureMeta.textContent = ` · ${pluralize(point.future_sites.length, "site")}`;
          futureSummary.appendChild(futureMeta);
          futureDetails.appendChild(futureSummary);
          const futureSection = document.createElement("section");
          futureSection.className = "data-section";
          const futureSurface = document.createElement("div");
          futureSurface.className = "data-section-surface";
          const futureList = document.createElement("div");
          futureList.className = "data-future-list";
          point.future_sites.forEach((site) => {
            futureList.appendChild(createFutureSiteItem(site, companyColor));
          });
          futureSurface.appendChild(futureList);
          futureSection.appendChild(futureSurface);
          futureDetails.appendChild(futureSection);
          pointFlow.appendChild(futureDetails);
        }

        return pointFlow;
      }

      function groupPointsByYear(points) {
        const groups = new Map();
        points
          .slice()
          .sort((a, b) => {
            const yearCompare = yearLabelForPoint(a).localeCompare(yearLabelForPoint(b));
            if (yearCompare !== 0) {
              return yearCompare;
            }
            const companyCompare = companySortIndex(a.company) - companySortIndex(b.company);
            if (companyCompare !== 0) {
              return companyCompare;
            }
            return String(a.date).localeCompare(String(b.date));
          })
          .forEach((point) => {
            const year = yearLabelForPoint(point);
            if (!groups.has(year)) {
              groups.set(year, []);
            }
            groups.get(year).push(point);
          });
        return Array.from(groups.entries()).map(([year, yearPoints]) => ({ year, points: yearPoints }));
      }

      function createYearGroup(group) {
        const wrap = document.createElement("section");
        wrap.className = "data-year-group";

        const yearLabel = document.createElement("h2");
        yearLabel.className = "data-year-label";
        yearLabel.textContent = group.year;
        wrap.appendChild(yearLabel);

        const thread = document.createElement("div");
        thread.className = "data-year-thread";
        group.points.forEach((point) => {
          thread.appendChild(createPointFlow(point));
        });
        wrap.appendChild(thread);

        return wrap;
      }

      function renderDataInspector(groups) {
        if (!dataInspector) {
          return;
        }
        dataInspector.textContent = "";
        if (!groups.length) {
          dataInspector.appendChild(createEmptyCard("No decomposition is available for the current selection."));
          return;
        }
        groups.forEach((group) => {
          dataInspector.appendChild(createYearGroup(group));
        });
      }

      function renderDataExplorer() {
        renderDataCompanyFilters();
        const points = dataPointsForCompany();
        const groups = groupPointsByYear(points);

        if (dataMeta) {
          if (!points.length) {
            dataMeta.textContent = "No points available";
          } else {
            dataMeta.textContent = `${groups.length} years · ${points.length} anchor points`;
          }
        }

        renderDataInspector(groups);
      }

      function updateActiveTab(nextTab) {
        if (!["chart", "data"].includes(nextTab) || state.activeTab === nextTab) {
          return state.activeTab;
        }
        if (nextTab === "data") {
          state.resumeAfterTab = state.playing;
          pause();
        }
        state.activeTab = nextTab;
        syncTabControls();
        if (nextTab === "data") {
          renderDataExplorer();
        } else {
          rebuild();
          if (state.resumeAfterTab) {
            state.resumeAfterTab = false;
            play();
          } else {
            renderFrame();
          }
        }
        return state.activeTab;
      }

      function updateDataCompany(nextCompany) {
        state.dataCompany = nextCompany;
        renderDataExplorer();
      }

      function syncRevealToggle() {
        if (!revealToggle) {
          return;
        }
        revealToggle.setAttribute("aria-pressed", state.revealMode ? "true" : "false");
        revealToggle.textContent = state.revealMode ? "Reveal on" : "Reveal off";
      }

      function updateRevealMode(nextRevealMode) {
        state.revealMode = Boolean(nextRevealMode);
        syncRevealToggle();
        renderFrame();
        return state.revealMode;
      }

      function setProgress(nextProgress) {
        state.progress = clamp(nextProgress, 0, 1);
        state.phasePauseActive = false;
        state.phasePauseElapsedMs = 0;
        state.phasePauseConsumed = state.progress >= PHASE_ONE_SHARE - 0.0001;
        state.lastTimestamp = null;
        renderFrame();
        return window.__trainingStory.getState();
      }

      function beginTimelineDrag() {
        if (state.dragging) {
          return;
        }
        state.dragging = true;
        state.resumeAfterDrag = state.playing;
        pause();
      }

      function endTimelineDrag() {
        if (!state.dragging) {
          return;
        }
        state.dragging = false;
        const shouldResume = state.resumeAfterDrag;
        state.resumeAfterDrag = false;
        if (shouldResume) {
          play();
        } else {
          syncPlaybackControls();
        }
      }

      const rebuild = () => {
        buildScene();
        renderFrame();
      };

      playToggle?.addEventListener("click", () => {
        if (state.playing) {
          pause();
        } else {
          play();
        }
      });

      replayButton?.addEventListener("click", () => {
        replay();
      });

      timelineRange?.addEventListener("pointerdown", () => {
        beginTimelineDrag();
      });

      timelineRange?.addEventListener("input", (event) => {
        const nextProgress = Number(event.target.value) / 1000;
        setProgress(nextProgress);
      });

      timelineRange?.addEventListener("change", () => {
        endTimelineDrag();
      });

      timelineRange?.addEventListener("pointerup", () => {
        endTimelineDrag();
      });

      timelineRange?.addEventListener("pointercancel", () => {
        endTimelineDrag();
      });

      timelineRange?.addEventListener("keydown", (event) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
          beginTimelineDrag();
        }
      });

      timelineRange?.addEventListener("keyup", (event) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
          endTimelineDrag();
        }
      });

      speedPresets?.addEventListener("click", (event) => {
        const button = event.target.closest(".speed-preset");
        if (!button?.dataset.speedStep) {
          return;
        }
        updateSpeed(button.dataset.speedStep);
      });

      revealToggle?.addEventListener("click", () => {
        updateRevealMode(!state.revealMode);
      });

      tabChart?.addEventListener("click", () => {
        updateActiveTab("chart");
      });

      tabData?.addEventListener("click", () => {
        updateActiveTab("data");
      });

      dataCompanyFilters?.addEventListener("click", (event) => {
        const button = event.target.closest(".data-pill");
        if (!button?.dataset.companyValue) {
          return;
        }
        updateDataCompany(button.dataset.companyValue);
      });

      dataDownload?.addEventListener("click", () => {
        downloadCurrentDataset();
      });

      new ResizeObserver(rebuild).observe(stage);
      window.addEventListener("resize", rebuild);

      window.__trainingStory = {
        play: () => {
          play();
          return window.__trainingStory.getState();
        },
        pause: () => {
          pause();
          return window.__trainingStory.getState();
        },
        replay: () => {
          replay();
          return window.__trainingStory.getState();
        },
        setProgress,
        setRevealMode: (nextRevealMode) => {
          updateRevealMode(nextRevealMode);
          return window.__trainingStory.getState();
        },
        getState: () => ({
          activeTab: state.activeTab,
          progress: state.progress,
          playing: state.playing,
          revealMode: state.revealMode,
          speedMultiplier: state.speedMultiplier,
          phasePauseActive: state.phasePauseActive,
          playheadTime: currentPlayheadTime(),
          formattedDate: formatDate(currentPlayheadTime()),
        }),
      };

      updateSpeed(state.speedStep);
      applyTheme(state.themeKey, { rebuild: false });
      syncTabControls();
      syncRevealToggle();
      syncPlaybackControls();
      syncTimelineControls();
      renderDataExplorer();
      rebuild();
      if (prefersReducedMotion) {
        pause();
      } else {
        play();
      }
    </script>
  </body>
</html>
"""
    return template.replace("__PAYLOAD__", serialized)


def main() -> None:
    paths.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    STORY_HTML.parent.mkdir(parents=True, exist_ok=True)
    rows = monthly.build_monthly_rows()
    overlays = monthly.read_model_overlays()
    story_html = html_template(build_story_payload(rows, overlays))
    STORY_HTML.write_text(story_html, encoding="utf-8")
    DOCS_STORY_HTML.write_text(story_html, encoding="utf-8")
    DOCS_INDEX_HTML.write_text(story_html, encoding="utf-8")


if __name__ == "__main__":
    main()
