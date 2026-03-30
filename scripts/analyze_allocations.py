#!/usr/bin/env python3

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Tuple

import paths

OUTPUT_DIR = paths.DERIVED_DIR
CURRENT_DATE = date(2026, 3, 27)
UNRESOLVED_USER = "Unresolved"


CONFIDENCE_RANK = {
    "confident": 3,
    "likely": 2,
    "speculative": 1,
    "unknown": 0,
}

SPLIT_RULES = {
    "Fluidstack Lake Mariner": {
        "building_users": ["G42", "G42", "Anthropic", "Anthropic", "Anthropic"],
        "rule_note": (
            "Buildings 1-2 are allocated to G42/Core42 and Buildings 3-5 to "
            "Anthropic based on the center notes and timeline row descriptions."
        ),
    }
}


@dataclass
class Party:
    company: str
    confidence: str
    raw: str


@dataclass
class TimelineRow:
    center: str
    snapshot_date: date
    buildings_operational: int
    power_mw: float
    h100_equivalents: float
    power_missing: bool
    h100_missing: bool
    status: str


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Optional[str]) -> float:
    if value is None:
        return 0.0
    cleaned = value.replace(",", "").strip()
    if cleaned == "":
        return 0.0
    return float(cleaned)


def parse_int(value: Optional[str]) -> int:
    return int(round(parse_float(value)))


def parse_parties(raw_value: str) -> List[Party]:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return []

    parties: List[Party] = []
    for raw_part in raw_value.split(","):
        raw_part = raw_part.strip()
        match = re.match(r"^(.*?)\s+#(\w+)$", raw_part)
        if match:
            company = match.group(1).strip()
            confidence = match.group(2).strip().lower()
        else:
            company = raw_part
            confidence = "unknown"
        parties.append(Party(company=company, confidence=confidence, raw=raw_part))
    return parties


def load_timeline_rows(path: Path) -> Dict[str, List[TimelineRow]]:
    rows_by_center: Dict[str, List[TimelineRow]] = defaultdict(list)
    for raw in read_csv(path):
        power_raw = (raw.get("Power (MW)") or "").strip()
        h100_raw = (raw.get("H100 equivalents") or "").strip()
        row = TimelineRow(
            center=raw["Data center"],
            snapshot_date=datetime.strptime(raw["Date"], "%Y-%m-%d").date(),
            buildings_operational=parse_int(raw["Buildings operational"]),
            power_mw=parse_float(power_raw),
            h100_equivalents=parse_float(h100_raw),
            power_missing=power_raw == "",
            h100_missing=h100_raw == "",
            status=(raw.get("Construction status") or "").strip(),
        )
        rows_by_center[row.center].append(row)

    for rows in rows_by_center.values():
        rows.sort(key=lambda row: row.snapshot_date)
        previous_power = 0.0
        previous_h100 = 0.0
        seen_power = False
        seen_h100 = False
        for row in rows:
            if row.power_missing and seen_power:
                row.power_mw = previous_power
            else:
                previous_power = row.power_mw
                seen_power = True

            if row.h100_missing and seen_h100:
                row.h100_equivalents = previous_h100
            else:
                previous_h100 = row.h100_equivalents
                seen_h100 = True
    return rows_by_center


def latest_row_as_of(rows: List[TimelineRow], as_of: date) -> Optional[TimelineRow]:
    latest: Optional[TimelineRow] = None
    for row in rows:
        if row.snapshot_date <= as_of:
            latest = row
        else:
            break
    return latest


def confidence_label(parties: List[Party]) -> str:
    if not parties:
        return "unknown"
    return parties[0].confidence


def build_issue(
    data_center: str,
    severity: str,
    issue_type: str,
    detail: str,
) -> dict:
    return {
        "data_center": data_center,
        "severity": severity,
        "issue_type": issue_type,
        "detail": detail,
    }


def build_center_mapping(center_row: dict) -> Tuple[dict, List[dict]]:
    name = center_row["Name"]
    owners = parse_parties(center_row.get("Owner", ""))
    users = parse_parties(center_row.get("Users", ""))
    issues: List[dict] = []

    mapping = {
        "data_center": name,
        "owner_raw": center_row.get("Owner", "").strip(),
        "users_raw": center_row.get("Users", "").strip(),
        "allocation_rule": "",
        "primary_user": "",
        "primary_user_confidence": "",
        "allocation_notes": "",
    }

    if name in SPLIT_RULES:
        mapping["allocation_rule"] = "split_by_building"
        mapping["primary_user"] = "MULTIPLE_PRIMARY_USERS"
        mapping["primary_user_confidence"] = "mixed"
        mapping["allocation_notes"] = SPLIT_RULES[name]["rule_note"]
        issues.append(
            build_issue(
                name,
                "medium",
                "multi_user_split_required",
                "Users field lists multiple users, so the site is split by building "
                "to stay MECE instead of allocating the entire center to one company.",
            )
        )
    elif not users:
        mapping["allocation_rule"] = "unresolved_no_user"
        mapping["primary_user"] = UNRESOLVED_USER
        mapping["primary_user_confidence"] = "unknown"
        mapping["allocation_notes"] = (
            "No primary user is listed in the dataset, so capacity stays in the "
            f"'{UNRESOLVED_USER}' bucket."
        )
        issues.append(
            build_issue(
                name,
                "high",
                "missing_user",
                "Users field is blank, so the center cannot be attributed to an end user "
                "without adding an external assumption.",
            )
        )
    elif len(users) == 1:
        mapping["allocation_rule"] = "single_user"
        mapping["primary_user"] = users[0].company
        mapping["primary_user_confidence"] = users[0].confidence
        mapping["allocation_notes"] = "Full center allocated to the single listed user."
    else:
        mapping["allocation_rule"] = "primary_user_first"
        mapping["primary_user"] = users[0].company
        mapping["primary_user_confidence"] = users[0].confidence
        mapping["allocation_notes"] = (
            "Users field contains multiple companies. Capacity is allocated to the "
            "first-listed user to stay MECE."
        )
        issues.append(
            build_issue(
                name,
                "high",
                "multi_user_primary_assumed",
                f"Users field lists {', '.join(p.company for p in users)}. "
                f"All capacity is assigned to {users[0].company} because the request "
                "requires a primary-only, non-overlapping allocation.",
            )
        )

    if not owners:
        issues.append(
            build_issue(
                name,
                "medium",
                "missing_owner",
                "Owner field is blank.",
            )
        )
    else:
        if owners[0].confidence == "speculative":
            issues.append(
                build_issue(
                    name,
                    "medium",
                    "speculative_owner",
                    f"Owner is only tagged as speculative: {owners[0].company}.",
                )
            )

    if users:
        highest_rank = max(CONFIDENCE_RANK.get(user.confidence, 0) for user in users)
        if highest_rank <= CONFIDENCE_RANK["speculative"]:
            issues.append(
                build_issue(
                    name,
                    "medium",
                    "speculative_user",
                    "Primary user evidence is speculative.",
                )
            )

    return mapping, issues


def allocate_split_center(
    center: str,
    rows: List[TimelineRow],
    as_of: date,
    issues: List[dict],
) -> List[dict]:
    rule = SPLIT_RULES[center]
    building_users = rule["building_users"]
    allocations: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"power_mw": 0.0, "h100_equivalents": 0.0}
    )

    previous_buildings = 0
    previous_power = 0.0
    previous_h100 = 0.0

    for row in rows:
        if row.snapshot_date > as_of:
            break

        delta_buildings = row.buildings_operational - previous_buildings
        delta_power = row.power_mw - previous_power
        delta_h100 = row.h100_equivalents - previous_h100

        if delta_buildings < 0:
            issues.append(
                build_issue(
                    center,
                    "high",
                    "split_rule_negative_building_delta",
                    f"Buildings operational decreased from {previous_buildings} to "
                    f"{row.buildings_operational} on {row.snapshot_date.isoformat()}.",
                )
            )
        elif delta_buildings > 0:
            new_building_users = building_users[
                previous_buildings : previous_buildings + delta_buildings
            ]
            if len(new_building_users) != delta_buildings:
                issues.append(
                    build_issue(
                        center,
                        "high",
                        "split_rule_missing_building_assignment",
                        f"Split rule only defines {len(building_users)} buildings, but "
                        f"the timeline reaches {row.buildings_operational}.",
                    )
                )
                unresolved_share = 1.0
                allocations[UNRESOLVED_USER]["power_mw"] += delta_power * unresolved_share
                allocations[UNRESOLVED_USER]["h100_equivalents"] += (
                    delta_h100 * unresolved_share
                )
            else:
                user_counts: Dict[str, int] = defaultdict(int)
                for user in new_building_users:
                    user_counts[user] += 1
                for user, count in user_counts.items():
                    share = count / delta_buildings
                    allocations[user]["power_mw"] += delta_power * share
                    allocations[user]["h100_equivalents"] += delta_h100 * share
        elif abs(delta_power) > 1e-9 or abs(delta_h100) > 1e-9:
            issues.append(
                build_issue(
                    center,
                    "medium",
                    "split_rule_non_building_capacity_change",
                    f"Capacity changed on {row.snapshot_date.isoformat()} without an "
                    "increase in buildings operational. Existing user split was left unchanged.",
                )
            )

        previous_buildings = row.buildings_operational
        previous_power = row.power_mw
        previous_h100 = row.h100_equivalents

    output_rows = []
    for user, metrics in sorted(allocations.items()):
        output_rows.append(
            {
                "allocated_user": user,
                "power_mw": metrics["power_mw"],
                "h100_equivalents": metrics["h100_equivalents"],
                "allocation_rule": "split_by_building",
            }
        )
    return output_rows


def allocate_center_snapshot(
    mapping: dict,
    center_rows: List[TimelineRow],
    as_of: date,
    issues: List[dict],
) -> List[dict]:
    if mapping["allocation_rule"] == "split_by_building":
        return allocate_split_center(mapping["data_center"], center_rows, as_of, issues)

    latest = latest_row_as_of(center_rows, as_of)
    power_mw = latest.power_mw if latest else 0.0
    h100_equivalents = latest.h100_equivalents if latest else 0.0

    return [
        {
            "allocated_user": mapping["primary_user"],
            "power_mw": power_mw,
            "h100_equivalents": h100_equivalents,
            "allocation_rule": mapping["allocation_rule"],
        }
    ]


def snapshot_specs(rows_by_center: Dict[str, List[TimelineRow]]) -> List[Tuple[str, date]]:
    max_year = max(
        row.snapshot_date.year for rows in rows_by_center.values() for row in rows
    )
    specs: List[Tuple[str, date]] = [("current", CURRENT_DATE)]
    for year in range(2024, max_year + 1):
        specs.append((f"year_end_{year}", date(year, 12, 31)))
    return specs


def round_metric(value: float) -> str:
    return f"{value:.6f}"


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_wide_table(
    rows: List[dict],
    metric: str,
    snapshot_filter_prefix: str = "year_end_",
) -> Tuple[List[str], List[dict]]:
    companies = sorted({row["company"] for row in rows if row["snapshot_label"].startswith(snapshot_filter_prefix)})
    years = sorted(
        {
            row["snapshot_label"].replace(snapshot_filter_prefix, "")
            for row in rows
            if row["snapshot_label"].startswith(snapshot_filter_prefix)
        }
    )

    table_rows = []
    for company in companies:
        entry = {"company": company}
        for year in years:
            value = 0.0
            for row in rows:
                if row["company"] == company and row["snapshot_label"] == f"{snapshot_filter_prefix}{year}":
                    value = row[metric]
                    break
            entry[year] = value
        table_rows.append(entry)

    return years, table_rows


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator] + body)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    centers = read_csv(paths.DATA_CENTERS_CSV)
    rows_by_center = load_timeline_rows(paths.DATA_CENTER_TIMELINES_CSV)

    mappings: List[dict] = []
    issues: List[dict] = []
    for center_row in centers:
        mapping, center_issues = build_center_mapping(center_row)
        mappings.append(mapping)
        issues.extend(center_issues)

    mappings_by_center = {mapping["data_center"]: mapping for mapping in mappings}

    for center, rows in rows_by_center.items():
        for row in rows:
            if row.power_missing:
                issues.append(
                    build_issue(
                        center,
                        "medium",
                        "missing_timeline_power",
                        f"Power is blank on {row.snapshot_date.isoformat()}, so the prior "
                        "known value is carried forward for snapshots.",
                    )
                )
            if row.h100_missing:
                issues.append(
                    build_issue(
                        center,
                        "medium",
                        "missing_timeline_h100",
                        f"H100 equivalents are blank on {row.snapshot_date.isoformat()}, so "
                        "the prior known value is carried forward for snapshots.",
                    )
                )
        previous_power = None
        previous_h100 = None
        for row in rows:
            if previous_power is not None and row.power_mw < previous_power - 1e-9:
                issues.append(
                    build_issue(
                        center,
                        "medium",
                        "non_monotonic_power_timeline",
                        f"Power drops from {previous_power:.1f} MW to {row.power_mw:.1f} MW "
                        f"on {row.snapshot_date.isoformat()}.",
                    )
                )
                break
            previous_power = row.power_mw
        for row in rows:
            if previous_h100 is not None and row.h100_equivalents < previous_h100 - 1e-9:
                issues.append(
                    build_issue(
                        center,
                        "medium",
                        "non_monotonic_h100_timeline",
                        f"H100 equivalents drop from {previous_h100:.0f} to "
                        f"{row.h100_equivalents:.0f} on {row.snapshot_date.isoformat()}.",
                    )
                )
                break
            previous_h100 = row.h100_equivalents

    allocation_rows: List[dict] = []
    company_rollups: Dict[Tuple[str, str], Dict[str, object]] = {}

    for snapshot_label, snapshot_date in snapshot_specs(rows_by_center):
        snapshot_total_allocated_power = 0.0
        snapshot_total_allocated_h100 = 0.0
        snapshot_total_source_power = 0.0
        snapshot_total_source_h100 = 0.0

        for center in sorted(rows_by_center):
            mapping = mappings_by_center[center]
            center_allocations = allocate_center_snapshot(
                mapping, rows_by_center[center], snapshot_date, issues
            )
            latest = latest_row_as_of(rows_by_center[center], snapshot_date)
            source_power = latest.power_mw if latest else 0.0
            source_h100 = latest.h100_equivalents if latest else 0.0

            allocated_power = 0.0
            allocated_h100 = 0.0
            for allocation in center_allocations:
                allocated_power += allocation["power_mw"]
                allocated_h100 += allocation["h100_equivalents"]
                allocation_rows.append(
                    {
                        "snapshot_label": snapshot_label,
                        "snapshot_date": snapshot_date.isoformat(),
                        "data_center": center,
                        "allocated_user": allocation["allocated_user"],
                        "power_mw": round_metric(allocation["power_mw"]),
                        "h100_equivalents": round_metric(allocation["h100_equivalents"]),
                        "allocation_rule": allocation["allocation_rule"],
                        "mapping_notes": mapping["allocation_notes"],
                    }
                )

                key = (snapshot_label, allocation["allocated_user"])
                if key not in company_rollups:
                    company_rollups[key] = {
                        "snapshot_label": snapshot_label,
                        "snapshot_date": snapshot_date.isoformat(),
                        "company": allocation["allocated_user"],
                        "power_mw": 0.0,
                        "h100_equivalents": 0.0,
                        "data_centers_count": 0,
                    }
                company_rollups[key]["power_mw"] += allocation["power_mw"]
                company_rollups[key]["h100_equivalents"] += allocation["h100_equivalents"]
                if allocation["power_mw"] > 0 or allocation["h100_equivalents"] > 0:
                    company_rollups[key]["data_centers_count"] += 1

            snapshot_total_allocated_power += allocated_power
            snapshot_total_allocated_h100 += allocated_h100
            snapshot_total_source_power += source_power
            snapshot_total_source_h100 += source_h100

            if abs(allocated_power - source_power) > 1e-6 or abs(allocated_h100 - source_h100) > 1e-6:
                issues.append(
                    build_issue(
                        center,
                        "high",
                        "allocation_total_mismatch",
                        f"Allocated totals ({allocated_power:.6f} MW, {allocated_h100:.6f} H100 eq) "
                        f"do not match the source snapshot ({source_power:.6f} MW, "
                        f"{source_h100:.6f} H100 eq) for {snapshot_label}.",
                    )
                )

        if abs(snapshot_total_allocated_power - snapshot_total_source_power) > 1e-6 or abs(
            snapshot_total_allocated_h100 - snapshot_total_source_h100
        ) > 1e-6:
            issues.append(
                build_issue(
                    "__all_centers__",
                    "high",
                    "snapshot_total_mismatch",
                    f"Allocated totals for {snapshot_label} do not match source totals.",
                )
            )

    company_rollup_rows: List[dict] = []
    for row in sorted(
        company_rollups.values(),
        key=lambda item: (item["snapshot_label"], item["company"]),
    ):
        company_rollup_rows.append(
            {
                "snapshot_label": row["snapshot_label"],
                "snapshot_date": row["snapshot_date"],
                "company": row["company"],
                "power_mw": round_metric(row["power_mw"]),
                "h100_equivalents": round_metric(row["h100_equivalents"]),
                "data_centers_count": row["data_centers_count"],
            }
        )

    current_totals = sorted(
        (
            row
            for row in company_rollups.values()
            if row["snapshot_label"] == "current"
        ),
        key=lambda item: item["h100_equivalents"],
        reverse=True,
    )

    yearly_headers, yearly_h100_rows = build_wide_table(
        [
            {
                "snapshot_label": row["snapshot_label"],
                "company": row["company"],
                "h100_equivalents": row["h100_equivalents"],
            }
            for row in company_rollups.values()
        ],
        metric="h100_equivalents",
    )

    markdown_current_rows = [
        [
            row["company"],
            f"{row['h100_equivalents']:.0f}",
            f"{row['power_mw']:.1f}",
            str(row["data_centers_count"]),
        ]
        for row in current_totals
        if row["h100_equivalents"] > 0
    ]

    markdown_year_rows = []
    for row in yearly_h100_rows:
        markdown_year_rows.append(
            [row["company"]] + [f"{row[year]:.0f}" for year in yearly_headers]
        )

    current_unresolved = next(
        (row for row in current_totals if row["company"] == UNRESOLVED_USER),
        None,
    )
    high_priority_issue_count = sum(1 for issue in issues if issue["severity"] == "high")
    medium_issue_count = sum(1 for issue in issues if issue["severity"] == "medium")

    summary_lines = [
        "# Primary User Compute Allocation",
        "",
        "## Method",
        "",
        f"- Current snapshot is taken as of {CURRENT_DATE.isoformat()} using the latest timeline row on or before that date.",
        "- Year-end snapshots use the latest timeline row on or before December 31 of each year.",
        "- Capacity is allocated to the listed user when a single user is present.",
        "- When multiple users are listed, capacity is assigned to the first-listed user unless the notes support an exact split.",
        f"- Centers with no listed user remain in the `{UNRESOLVED_USER}` bucket rather than being forced onto an owner.",
        "",
        "## Current Snapshot",
        "",
        format_markdown_table(
            ["Company", "H100 eq", "Power MW", "Centers"],
            markdown_current_rows,
        ),
        "",
        "## Year-End H100 Equivalents By Primary User",
        "",
        format_markdown_table(["Company"] + yearly_headers, markdown_year_rows),
        "",
        "## Key Issues",
        "",
        f"- High-severity issues: {high_priority_issue_count}",
        f"- Medium-severity issues: {medium_issue_count}",
        (
            f"- Current unresolved capacity: {current_unresolved['h100_equivalents']:.0f} H100 eq "
            f"across {current_unresolved['data_centers_count']} centers."
            if current_unresolved
            else "- Current unresolved capacity: 0 H100 eq."
        ),
        "- `Fluidstack Lake Mariner` is split exactly by building across G42 and Anthropic because the notes provide a user-by-building mapping.",
        "- `Microsoft Fairwater Wisconsin` is allocated entirely to OpenAI because the dataset lists `OpenAI` first in a multi-user field, but this remains a material ambiguity.",
        "- `xAI Colossus 1` has a non-monotonic timeline, so trend work should use latest snapshots rather than cumulative maxima.",
        "",
        "## Output Files",
        "",
        "- `data/derived/data_center_primary_user_mapping.csv`",
        "- `data/derived/data_center_allocations_by_snapshot.csv`",
        "- `data/derived/company_capacity_by_snapshot.csv`",
        "- `data/derived/allocation_issues.csv`",
    ]

    write_csv(
        paths.PRIMARY_USER_MAPPING_CSV,
        [
            "data_center",
            "owner_raw",
            "users_raw",
            "allocation_rule",
            "primary_user",
            "primary_user_confidence",
            "allocation_notes",
        ],
        mappings,
    )
    write_csv(
        paths.ALLOCATIONS_BY_SNAPSHOT_CSV,
        [
            "snapshot_label",
            "snapshot_date",
            "data_center",
            "allocated_user",
            "power_mw",
            "h100_equivalents",
            "allocation_rule",
            "mapping_notes",
        ],
        allocation_rows,
    )
    write_csv(
        paths.COMPANY_CAPACITY_BY_SNAPSHOT_CSV,
        [
            "snapshot_label",
            "snapshot_date",
            "company",
            "power_mw",
            "h100_equivalents",
            "data_centers_count",
        ],
        company_rollup_rows,
    )
    write_csv(
        paths.ALLOCATION_ISSUES_CSV,
        ["data_center", "severity", "issue_type", "detail"],
        issues,
    )
    paths.ALLOCATION_SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
